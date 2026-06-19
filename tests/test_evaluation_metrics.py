"""Unit tests for VeraRAG evaluation metrics."""

import sys

sys.path.insert(0, 'src')

import unittest

from src.evaluation.answer_metrics import AnswerMetrics
from src.evaluation.calibration_metrics import CalibrationMetrics
from src.evaluation.conflict_metrics import ConflictMetrics
from src.evaluation.evidence_metrics import EvidenceMetrics
from src.evaluation.hallucination_metrics import HallucinationMetrics
from src.utils.data_structures import ConflictType


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

    def test_f1_score_counts_duplicate_tokens(self):
        """Repeated predicted tokens should lower precision instead of being de-duped."""
        self.assertAlmostEqual(AnswerMetrics.f1_score("Paris Paris", "Paris"), 2 / 3)

    def test_f1_score_handles_empty_answers(self):
        self.assertEqual(AnswerMetrics.f1_score("", ""), 1.0)
        self.assertEqual(AnswerMetrics.f1_score("", "Paris"), 0.0)

    def test_soft_f1_accepts_reference_inside_complete_chinese_answer(self):
        score = AnswerMetrics.soft_f1_score(
            "中国生成式AI管理暂行办法于2023年8月15日正式施行。",
            "2023年8月15日。",
        )

        self.assertEqual(score, 1.0)

    def test_soft_f1_handles_chinese_paraphrase_without_whitespace(self):
        score = AnswerMetrics.soft_f1_score(
            "谷歌发布的Willow处理器拥有105个量子比特。",
            "Willow的量子比特数量是105个。",
        )

        self.assertGreater(score, 0.3)

    def test_soft_f1_handles_empty_inputs_and_single_chinese_chars(self):
        self.assertEqual(AnswerMetrics.soft_f1_score("", ""), 1.0)
        self.assertEqual(AnswerMetrics.soft_f1_score("", "巴黎"), 0.0)
        self.assertEqual(AnswerMetrics.soft_f1_score("猫", "狗"), 0.0)

    def test_batch_answer_metrics_validate_lengths_and_empty_batches(self):
        self.assertEqual(AnswerMetrics.batch_exact_match([], []), 0.0)
        self.assertEqual(AnswerMetrics.batch_f1_score([], []), 0.0)
        self.assertEqual(AnswerMetrics.batch_soft_f1_score([], []), 0.0)

        with self.assertRaisesRegex(ValueError, "same length"):
            AnswerMetrics.batch_exact_match(["a"], [])
        with self.assertRaisesRegex(ValueError, "same length"):
            AnswerMetrics.batch_f1_score(["a"], [])
        with self.assertRaisesRegex(ValueError, "same length"):
            AnswerMetrics.batch_soft_f1_score(["a"], [])

    def test_batch_soft_f1_and_compute_all_include_versioned_soft_metric(self):
        self.assertEqual(
            AnswerMetrics.batch_soft_f1_score(
                ["中国生成式AI管理暂行办法于2023年8月15日正式施行。"],
                ["2023年8月15日。"],
            ),
            1.0,
        )

        metrics = AnswerMetrics.compute_all("Paris Paris", "Paris")

        self.assertEqual(metrics["exact_match"], 0.0)
        self.assertAlmostEqual(metrics["f1_score"], 2 / 3)
        self.assertEqual(metrics["soft_f1_score"], 1.0)


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

    def test_supporting_fact_and_citation_metrics(self):
        """Test supporting fact, joint, and citation metrics."""
        self.assertEqual(EvidenceMetrics.supporting_fact_precision(["s1", "s2"], ["s1"]), 0.5)
        self.assertEqual(EvidenceMetrics.supporting_fact_recall(["s1"], ["s1", "s2"]), 0.5)
        self.assertEqual(EvidenceMetrics.supporting_fact_f1(["s1"], ["s1", "s2"]), 2 / 3)
        self.assertEqual(EvidenceMetrics.joint_em(1.0, 1.0), 1.0)
        self.assertEqual(EvidenceMetrics.joint_em(1.0, 0.0), 0.0)
        self.assertEqual(
            EvidenceMetrics.citation_precision("Claim [E1] and unsupported [E9]", {}, {"E1"}),
            0.5,
        )
        self.assertEqual(EvidenceMetrics.citation_recall("Claim [E1]", {"E1", "E2"}), 0.5)


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

    def test_conflict_type_accuracy_counts_missing_gold_pairs_as_errors(self):
        """Missing predicted conflict types should not inflate type accuracy."""
        accuracy = ConflictMetrics.conflict_type_accuracy(
            [(("B", "A"), ConflictType.NUMERIC_CONFLICT)],
            [
                (("A", "B"), ConflictType.NUMERIC_CONFLICT),
                (("C", "D"), ConflictType.TEMPORAL_CONFLICT),
            ],
        )

        self.assertEqual(accuracy, 0.5)

    def test_conflict_resolution_accuracy_and_compute_all(self):
        """Test resolution accuracy and aggregate conflict metrics."""
        accuracy = ConflictMetrics.conflict_resolution_accuracy(
            [{"conflict_pair": ("A", "B"), "resolution": "prefer_newer"}],
            [
                {"conflict_pair": ("A", "B"), "resolution": "prefer_newer"},
                {"conflict_pair": ("C", "D"), "resolution": "abstain"},
            ],
        )
        self.assertEqual(accuracy, 0.5)

        metrics = ConflictMetrics.compute_all(
            [("A", "B")],
            [("B", "A")],
            [(("A", "B"), ConflictType.REFUTE)],
            [(("B", "A"), ConflictType.REFUTE)],
        )

        self.assertEqual(metrics["conflict_detection_f1"], 1.0)
        self.assertEqual(metrics["conflict_type_accuracy"], 1.0)


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

    def test_ece_includes_confidence_one_in_final_bin(self):
        """A confidence of exactly 1.0 must be included in the final ECE bin."""
        ece = CalibrationMetrics.expected_calibration_error([1.0], [False], n_bins=10)

        self.assertEqual(ece, 1.0)

    def test_auroc_risk_curve_and_compute_all(self):
        """Test aggregate calibration helpers."""
        auroc = CalibrationMetrics.compute_auroc_abstention(
            [0.1, 0.9],
            [False, True],
            abstention_thresholds=5,
        )
        self.assertGreater(auroc, 0.9)

        curve = CalibrationMetrics.risk_coverage_curve(
            [0.9, 0.2, 0.8],
            [True, False, False],
            n_points=3,
        )
        self.assertEqual(curve["coverage"], [1 / 3, 2 / 3, 1.0])
        self.assertEqual(curve["error_rate"][:2], [0.0, 0.5])
        self.assertAlmostEqual(curve["error_rate"][2], 2 / 3)

        metrics = CalibrationMetrics.compute_all([1.0, 0.0], [True, False])
        self.assertEqual(metrics["brier_score"], 0.0)


class TestHallucinationMetrics(unittest.TestCase):
    """Test hallucination metrics."""

    def test_unsupported_claim_rate(self):
        """Test unsupported claim rate."""
        claims = ["RAG helps", "RAG is perfect", "RAG has limits"]
        supported = {"RAG helps", "RAG has limits"}

        rate = HallucinationMetrics.unsupported_claim_rate(claims, supported)
        self.assertEqual(rate, 1/3)

    def test_unsupported_claim_rate_normalizes_claim_text(self):
        claims = ["  RAG helps ", "RAG is perfect"]
        supported = {"rag helps"}

        rate = HallucinationMetrics.unsupported_claim_rate(claims, supported)

        self.assertEqual(rate, 0.5)

    def test_empty_hallucination_inputs_return_zero(self):
        self.assertEqual(HallucinationMetrics.unsupported_claim_rate([], {"supported"}), 0.0)
        self.assertEqual(HallucinationMetrics.entity_hallucination_rate(set(), {"Apple"}), 0.0)
        self.assertEqual(HallucinationMetrics.numerical_hallucination_rate([], {1.0}), 0.0)
        self.assertEqual(HallucinationMetrics.overclaiming_rate([0.1, 0.2], [False, True]), 0.0)

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

    def test_extract_numbers_preserves_signs_and_ranges(self):
        text = "Revenue fell -5.5%, margin rose +2%, and range was 2023-2024 with loss -1,200."

        numbers = HallucinationMetrics.extract_numbers_from_text(text)

        self.assertIn(-0.055, numbers)
        self.assertIn(0.02, numbers)
        self.assertIn(2023.0, numbers)
        self.assertIn(2024.0, numbers)
        self.assertIn(-1200.0, numbers)

    def test_hallucination_rates_and_compute_all(self):
        """Test aggregate hallucination metric behavior."""
        self.assertEqual(
            HallucinationMetrics.entity_hallucination_rate({"Apple", "Meta"}, {"Apple"}),
            0.5,
        )
        self.assertEqual(
            HallucinationMetrics.numerical_hallucination_rate([100.0, 103.0], {100.5}, tolerance=0.01),
            0.5,
        )
        self.assertEqual(
            HallucinationMetrics.overclaiming_rate([0.9, 0.7, 0.95], [False, False, True]),
            0.5,
        )
        self.assertEqual(
            HallucinationMetrics.extract_claims_from_text("Short. VeraRAG cites evidence. It abstains when unsupported."),
            ["verarag cites evidence", "it abstains when unsupported"],
        )

        metrics = HallucinationMetrics.compute_all(
            "Apple reported 85% accuracy.",
            ["Apple reported 85% accuracy."],
            [0.9],
            [True],
        )

        self.assertEqual(metrics["entity_hallucination_rate"], 0.0)
        self.assertEqual(metrics["numerical_hallucination_rate"], 0.0)

    def test_signed_numbers_affect_numerical_hallucination(self):
        self.assertEqual(
            HallucinationMetrics.numerical_hallucination_rate([-0.05], {0.05}),
            1.0,
        )
        self.assertEqual(
            HallucinationMetrics.numerical_hallucination_rate([-0.05], {-0.0501}, tolerance=0.01),
            0.0,
        )

        metrics = HallucinationMetrics.compute_all(
            "Acme revenue fell -5%.",
            ["Acme revenue rose 5%."],
            [0.9],
            [False],
        )

        self.assertEqual(metrics["numerical_hallucination_rate"], 1.0)

    def test_overclaiming_rate_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            HallucinationMetrics.overclaiming_rate([0.9], [True, False])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            HallucinationMetrics.overclaiming_rate([0.9], [True], confidence_threshold=1.5)
        with self.assertRaisesRegex(ValueError, "finite values"):
            HallucinationMetrics.overclaiming_rate([float("nan")], [True])

    def test_numerical_hallucination_rate_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "Tolerance"):
            HallucinationMetrics.numerical_hallucination_rate([1.0], {1.0}, tolerance=-0.1)
        with self.assertRaisesRegex(ValueError, "Answer numbers"):
            HallucinationMetrics.numerical_hallucination_rate([float("inf")], {1.0})
        with self.assertRaisesRegex(ValueError, "Evidence numbers"):
            HallucinationMetrics.numerical_hallucination_rate([1.0], {float("nan")})


if __name__ == "__main__":
    unittest.main()
