"""Answer Metrics for VeraRAG Evaluation."""

from typing import Dict, Any, List, Optional
import re


class AnswerMetrics:
    """
    Metrics for evaluating answer correctness.

    Includes:
    - Exact Match (EM)
    - F1 Score
    - Token-level accuracy
    """

    @staticmethod
    def exact_match(predicted: str, reference: str) -> float:
        """
        Calculate exact match score.

        Args:
            predicted: Predicted answer
            reference: Reference/ground truth answer

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Normalize whitespace and case
        pred_normalized = AnswerMetrics._normalize_answer(predicted)
        ref_normalized = AnswerMetrics._normalize_answer(reference)

        return 1.0 if pred_normalized == ref_normalized else 0.0

    @staticmethod
    def f1_score(predicted: str, reference: str) -> float:
        """
        Calculate F1 score between predicted and reference answers.

        Args:
            predicted: Predicted answer
            reference: Reference/ground truth answer

        Returns:
            F1 score (0-1)
        """
        pred_tokens = AnswerMetrics._tokenize(predicted)
        ref_tokens = AnswerMetrics._tokenize(reference)

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        common = set(pred_tokens) & set(ref_tokens)

        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def batch_exact_match(predictions: List[str], references: List[str]) -> float:
        """Calculate average exact match across a batch."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        scores = [AnswerMetrics.exact_match(p, r) for p, r in zip(predictions, references)]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def batch_f1_score(predictions: List[str], references: List[str]) -> float:
        """Calculate average F1 score across a batch."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        scores = [AnswerMetrics.f1_score(p, r) for p, r in zip(predictions, references)]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        # Lowercase
        answer = answer.lower()
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', ' ', answer)
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        return answer

    @staticmethod
    def _tokenize(answer: str) -> List[str]:
        """Tokenize answer into words."""
        return AnswerMetrics._normalize_answer(answer).split()

    @staticmethod
    def compute_all(
        predicted: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Compute all answer metrics.

        Args:
            predicted: Predicted answer
            reference: Reference/ground truth answer

        Returns:
            Dictionary with metric values
        """
        return {
            "exact_match": AnswerMetrics.exact_match(predicted, reference),
            "f1_score": AnswerMetrics.f1_score(predicted, reference)
        }
