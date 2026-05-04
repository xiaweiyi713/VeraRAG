"""Calibration Metrics for VeraRAG Evaluation."""

from typing import Dict, Any, List
import numpy as np


class CalibrationMetrics:
    """
    Metrics for evaluating confidence calibration.

    Includes:
    - Expected Calibration Error (ECE)
    - Brier Score
    - AUROC for Abstention
    - Risk-Coverage Curve
    """

    @staticmethod
    def expected_calibration_error(
        confidences: List[float],
        correct: List[bool],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures the difference between predicted confidence
        and actual accuracy across confidence bins.

        Args:
            confidences: List of confidence scores (0-1)
            correct: List of correctness indicators
            n_bins: Number of bins for ECE calculation

        Returns:
            ECE score (0-1, lower is better)
        """
        if not confidences:
            return 0.0

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_weight = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [
                (c, r) for c, r in zip(confidences, correct)
                if bin_lower <= c < bin_upper
            ]

            if not in_bin:
                continue

            bin_confidence = sum(c for c, _ in in_bin) / len(in_bin)
            bin_accuracy = sum(1 for _, r in in_bin if r) / len(in_bin)
            bin_weight = len(in_bin) / len(confidences)

            ece += bin_weight * abs(bin_confidence - bin_accuracy)
            total_weight += bin_weight

        return ece if total_weight > 0 else 0.0

    @staticmethod
    def brier_score(
        confidences: List[float],
        correct: List[bool]
    ) -> float:
        """
        Calculate Brier Score.

        Brier score measures the mean squared error between
        predicted probabilities and outcomes.

        Args:
            confidences: List of confidence scores (0-1)
            correct: List of correctness indicators

        Returns:
            Brier score (0-1, lower is better)
        """
        if not confidences:
            return 0.0

        labels = [1.0 if c else 0.0 for c in correct]

        brier = sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)

        return brier

    @staticmethod
    def compute_auroc_abstention(
        confidences: List[float],
        correct: List[bool],
        abstention_thresholds: int = 100
    ) -> float:
        """
        Calculate AUROC for abstention decisions.

        Measures how well the system can identify when to abstain.

        Args:
            confidences: List of confidence scores (0-1)
            correct: List of correctness indicators
            abstention_thresholds: Number of thresholds to evaluate

        Returns:
            AUROC score (0-1, higher is better)
        """
        if not confidences:
            return 0.0

        # Sort by confidence
        sorted_data = sorted(zip(confidences, correct), key=lambda x: x[0])

        # Calculate abstention benefit at each threshold
        thresholds = np.linspace(0, 1, abstention_thresholds)

        tpr_list = []  # True positive rate (abstain on incorrect)
        fpr_list = []  # False positive rate (abstain on correct)

        for threshold in thresholds:
            tp = sum(1 for c, corr in sorted_data if c < threshold and not corr)
            fp = sum(1 for c, corr in sorted_data if c < threshold and corr)
            tn = sum(1 for c, corr in sorted_data if c >= threshold and corr)
            fn = sum(1 for c, corr in sorted_data if c >= threshold and not corr)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Calculate AUROC using trapezoidal rule
        auroc = 0.0
        for i in range(len(fpr_list) - 1):
            auroc += (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i] + tpr_list[i + 1]) / 2

        return auroc

    @staticmethod
    def risk_coverage_curve(
        confidences: List[float],
        correct: List[bool],
        n_points: int = 20
    ) -> Dict[str, List[float]]:
        """
        Calculate risk-coverage curve data.

        Shows trade-off between coverage (fraction answered)
        and error rate (fraction incorrect among answered).

        Args:
            confidences: List of confidence scores (0-1)
            correct: List of correctness indicators
            n_points: Number of points on the curve

        Returns:
            Dictionary with 'coverage' and 'error_rate' lists
        """
        if not confidences:
            return {"coverage": [], "error_rate": []}

        # Sort by confidence (descending)
        sorted_data = sorted(zip(confidences, correct), key=lambda x: -x[0])

        coverage = []
        error_rates = []

        for i in range(1, n_points + 1):
            threshold = i / n_points
            n_answered = int(len(confidences) * threshold)

            if n_answered == 0:
                coverage.append(0.0)
                error_rates.append(0.0)
                continue

            answered = sorted_data[:n_answered]
            n_correct = sum(1 for _, c in answered if c)

            cov = n_answered / len(confidences)
            err = 1.0 - (n_correct / n_answered) if n_answered > 0 else 0.0

            coverage.append(cov)
            error_rates.append(err)

        return {"coverage": coverage, "error_rate": error_rates}

    @staticmethod
    def compute_all(
        confidences: List[float],
        correct: List[bool],
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute all calibration metrics.

        Args:
            confidences: List of confidence scores (0-1)
            correct: List of correctness indicators
            n_bins: Number of bins for ECE

        Returns:
            Dictionary with metric values
        """
        return {
            "expected_calibration_error": CalibrationMetrics.expected_calibration_error(
                confidences, correct, n_bins
            ),
            "brier_score": CalibrationMetrics.brier_score(confidences, correct),
            "auroc_abstention": CalibrationMetrics.compute_auroc_abstention(
                confidences, correct
            )
        }
