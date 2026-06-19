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
        predictions = np.asarray(predictions, dtype=float)
        labels = np.asarray(labels, dtype=float)
        self._validate_temperature_inputs(predictions, labels, max_iter, lr)

        # Initialize temperature
        temperature = 1.0

        # Convert to probabilities if needed
        if np.all((predictions >= 0.0) & (predictions <= 1.0)):
            clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped))
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
            if correct:
                raise ValueError("confidences and correct must have the same length")
            return {"ece": 0.0, "brier_score": 0.0}

        self._validate_metric_inputs(confidences, correct)

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
        for index, (bin_lower, bin_upper) in enumerate(
            zip(bin_lowers, bin_uppers, strict=True)
        ):
            if index == n_bins - 1:
                in_bin = [
                    (c, r) for c, r in zip(confidences, correct, strict=True)
                    if bin_lower <= c <= bin_upper
                ]
            else:
                in_bin = [
                    (c, r) for c, r in zip(confidences, correct, strict=True)
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

    @staticmethod
    def _validate_temperature_inputs(
        predictions: np.ndarray,
        labels: np.ndarray,
        max_iter: int,
        lr: float,
    ) -> None:
        """Validate inputs before numeric temperature optimization."""
        if predictions.shape != labels.shape:
            raise ValueError("predictions and labels must have the same shape")
        if predictions.size == 0:
            raise ValueError("temperature calibration requires at least one sample")
        if max_iter < 0:
            raise ValueError("max_iter must be non-negative")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not np.all(np.isfinite(predictions)):
            raise ValueError("predictions must be finite")
        if not np.all(np.isfinite(labels)):
            raise ValueError("labels must be finite")
        if not np.all((labels == 0.0) | (labels == 1.0)):
            raise ValueError("labels must be binary 0/1 values")

    @staticmethod
    def _validate_metric_inputs(
        confidences: list[float],
        correct: list[bool],
    ) -> None:
        """Validate confidence/correctness vectors for calibration metrics."""
        if len(confidences) != len(correct):
            raise ValueError("confidences and correct must have the same length")
        for confidence in confidences:
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidences must be finite values in [0, 1]")
        for is_correct in correct:
            if not isinstance(is_correct, bool | np.bool_):
                raise ValueError("correct values must be booleans")
