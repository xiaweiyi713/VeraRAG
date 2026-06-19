"""Answer Metrics for VeraRAG Evaluation."""

import re
from collections import Counter


class AnswerMetrics:
    """
    Metrics for evaluating answer correctness.

    Includes:
    - Exact Match (EM)
    - Token-level F1 Score
    - Soft F1 (keyword/number overlap, better for Chinese free-text)
    """

    VERSION = "soft-f1-v2"

    @staticmethod
    def exact_match(predicted: str, reference: str) -> float:
        pred_normalized = AnswerMetrics._normalize_answer(predicted)
        ref_normalized = AnswerMetrics._normalize_answer(reference)
        return 1.0 if pred_normalized == ref_normalized else 0.0

    @staticmethod
    def f1_score(predicted: str, reference: str) -> float:
        return AnswerMetrics._multiset_f1(
            AnswerMetrics._tokenize(predicted),
            AnswerMetrics._tokenize(reference),
        )

    @staticmethod
    def soft_f1_score(predicted: str, reference: str) -> float:
        """Keyword/number overlap F1 for Chinese free-text answers.

        Uses normalized containment for concise fact answers and combines
        keyword overlap with Chinese character-bigram overlap for longer text.
        """
        pred_normalized = AnswerMetrics._compact_normalize(predicted)
        ref_normalized = AnswerMetrics._compact_normalize(reference)
        if not pred_normalized and not ref_normalized:
            return 1.0
        if not pred_normalized or not ref_normalized:
            return 0.0
        if ref_normalized in pred_normalized:
            return 1.0

        pred_keys = AnswerMetrics._extract_keywords(predicted)
        ref_keys = AnswerMetrics._extract_keywords(reference)
        keyword_f1 = AnswerMetrics._multiset_f1(pred_keys, ref_keys) if pred_keys or ref_keys else 0.0
        cjk_f1 = AnswerMetrics._multiset_f1(
            AnswerMetrics._soft_tokens(predicted),
            AnswerMetrics._soft_tokens(reference),
        )
        return max(keyword_f1, cjk_f1)

    @staticmethod
    def _multiset_f1(predicted: list[str], reference: list[str]) -> float:
        if not predicted and not reference:
            return 1.0
        if not predicted or not reference:
            return 0.0
        common = sum((Counter(predicted) & Counter(reference)).values())
        precision = common / len(predicted)
        recall = common / len(reference)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _soft_tokens(text: str) -> list[str]:
        """Tokenize mixed Chinese/English text without external dependencies."""
        normalized = AnswerMetrics._normalize_answer(text)
        tokens = re.findall(r"\d+(?:\.\d+)?%?|[a-z]+(?:-[a-z]+)*", normalized)
        chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
        for run in chinese_runs:
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
        return tokens

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract keywords: numbers, English words, Chinese segments (2+ chars)."""
        keywords = []

        # Numbers (including decimals and percentages)
        numbers = re.findall(r'\d+\.?\d*%?', text)
        keywords.extend(numbers)

        # English words
        english = re.findall(r'[A-Za-z]+(?:[-][A-Za-z]+)*', text)
        keywords.extend(w.lower() for w in english if len(w) > 1)

        # Chinese key phrases: split by punctuation, take segments 2-10 chars
        chinese_segments = re.split(r'[，。、；：！？\s,;:!?\(\)（）【】《》""'r'\-/]', text)
        for seg in chinese_segments:
            seg = seg.strip()
            if 2 <= len(seg) <= 20:
                keywords.append(seg)
            elif len(seg) > 20:
                # Take character bigrams for long segments
                for i in range(0, min(len(seg), 40) - 1):
                    keywords.append(seg[i:i + 2])

        return keywords

    @staticmethod
    def batch_exact_match(predictions: list[str], references: list[str]) -> float:
        """Calculate average exact match across a batch."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        scores = [
            AnswerMetrics.exact_match(p, r)
            for p, r in zip(predictions, references, strict=False)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def batch_f1_score(predictions: list[str], references: list[str]) -> float:
        """Calculate average F1 score across a batch."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        scores = [
            AnswerMetrics.f1_score(p, r)
            for p, r in zip(predictions, references, strict=False)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def batch_soft_f1_score(predictions: list[str], references: list[str]) -> float:
        """Calculate average soft F1 score across a batch."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        scores = [
            AnswerMetrics.soft_f1_score(p, r)
            for p, r in zip(predictions, references, strict=False)
        ]
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
    def _compact_normalize(answer: str) -> str:
        return re.sub(r"\s+", "", AnswerMetrics._normalize_answer(answer))

    @staticmethod
    def _tokenize(answer: str) -> list[str]:
        """Tokenize answer into words."""
        return AnswerMetrics._normalize_answer(answer).split()

    @staticmethod
    def compute_all(
        predicted: str,
        reference: str
    ) -> dict[str, float]:
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
            "f1_score": AnswerMetrics.f1_score(predicted, reference),
            "soft_f1_score": AnswerMetrics.soft_f1_score(predicted, reference),
        }
