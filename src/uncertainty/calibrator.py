"""Confidence Calibrator for VeraRAG."""

from typing import Any

import numpy as np

from ..utils.data_structures import UncertaintyBreakdown


class ConfidenceCalibrator:
    """
    Calibrates confidence scores to be well-calibrated.

    A well-calibrated model's predicted confidence should match
    the actual accuracy (e.g., predictions with 80% confidence
    should be correct 80% of the time).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.calibration_data: list[tuple[float, bool]] = []
        self.is_calibrated = False

    def add_sample(self, confidence: float, is_correct: bool) -> None:
        """
        Add a calibration sample.

        Args:
            confidence: Predicted confidence (0-1)
            is_correct: Whether the prediction was correct
        """
        self.calibration_data.append((confidence, is_correct))
        self.is_calibrated = False

    def calibrate_temperature(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 1000,
        lr: float = 0.01
    ) -> float:
        """
        Find optimal temperature scaling for calibration.

        Args:
            predictions: Logits or confidence scores
            labels: Binary labels (0 or 1)
            max_iter: Maximum optimization iterations
            lr: Learning rate

        Returns:
            Optimal temperature value
        """
        # Initialize temperature
        temperature = 1.0

        # Convert to probabilities if needed
        if predictions.max() <= 1.0:
            logits = np.log(predictions / (1 - predictions + 1e-10))
        else:
            logits = predictions

        for _ in range(max_iter):
            # Apply temperature scaling
            scaled = logits / temperature
            probs = 1 / (1 + np.exp(-scaled))

            # Compute gradient
            gradient = np.mean((probs - labels) * logits * (1 - probs) * probs / temperature)

            # Update temperature
            temperature -= lr * gradient
            temperature = max(0.1, min(10.0, temperature))

        return temperature

    def calibrate_confidence(
        self,
        raw_confidence: float,
        uncertainty: UncertaintyBreakdown
    ) -> float:
        """
        Calibrate a raw confidence score based on uncertainty.

        Args:
            raw_confidence: Raw confidence from the model
            uncertainty: Uncertainty breakdown

        Returns:
            Calibrated confidence score
        """
        # Start with raw confidence
        calibrated = raw_confidence

        # Reduce based on uncertainty components
        # High retrieval uncertainty -> reduce confidence
        calibrated *= (1.0 - uncertainty.retrieval_uncertainty * 0.3)

        # High conflict -> reduce confidence
        calibrated *= (1.0 - uncertainty.evidence_conflict * 0.4)

        # High reasoning gap -> reduce confidence
        calibrated *= (1.0 - uncertainty.reasoning_gap * 0.3)

        # Low source reliability -> reduce confidence
        calibrated *= (1.0 - uncertainty.source_reliability * 0.2)

        # Verification uncertainty -> reduce confidence
        calibrated *= (1.0 - uncertainty.verification_uncertainty * 0.5)

        return max(0.0, min(1.0, calibrated))

    def compute_calibration_metrics(
        self,
        confidences: list[float],
        correct: list[bool]
    ) -> dict[str, float]:
        """
        Compute calibration metrics.

        Args:
            confidences: List of confidence scores
            correct: List of correctness indicators

        Returns:
            Dictionary with calibration metrics
        """
        if not confidences:
            return {"ece": 0.0, "brier_score": 0.0}

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(confidences, correct)

        # Brier Score
        brier = self._compute_brier_score(confidences, correct)

        return {
            "ece": ece,
            "brier_score": brier
        }

    def _compute_ece(
        self,
        confidences: list[float],
        correct: list[bool],
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):  # noqa: B905
            in_bin = [
                (c, r) for c, r in zip(confidences, correct)  # noqa: B905
                if bin_lower <= c < bin_upper
            ]

            if not in_bin:
                continue

            bin_confidence = sum(c for c, _ in in_bin) / len(in_bin)
            bin_accuracy = sum(1 for _, r in in_bin if r) / len(in_bin)
            bin_weight = len(in_bin) / len(confidences)

            ece += bin_weight * abs(bin_confidence - bin_accuracy)

        return ece

    def _compute_brier_score(
        self,
        confidences: list[float],
        correct: list[bool]
    ) -> float:
        """Compute Brier Score."""
        labels = [1.0 if c else 0.0 for c in correct]

        brier = sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)  # noqa: E741, B905

        return brier
