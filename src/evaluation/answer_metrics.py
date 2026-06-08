"""Answer Metrics for VeraRAG Evaluation."""

import re


class AnswerMetrics:
    """
    Metrics for evaluating answer correctness.

    Includes:
    - Exact Match (EM)
    - Token-level F1 Score
    - Soft F1 (keyword/number overlap, better for Chinese free-text)
    """

    @staticmethod
    def exact_match(predicted: str, reference: str) -> float:
        pred_normalized = AnswerMetrics._normalize_answer(predicted)
        ref_normalized = AnswerMetrics._normalize_answer(reference)
        return 1.0 if pred_normalized == ref_normalized else 0.0

    @staticmethod
    def f1_score(predicted: str, reference: str) -> float:
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
    def soft_f1_score(predicted: str, reference: str) -> float:
        """Keyword/number overlap F1 for Chinese free-text answers.

        Extracts key entities (numbers, proper nouns, domain terms) from both
        answers and computes overlap. More lenient than token-level F1.
        """
        pred_keys = AnswerMetrics._extract_keywords(predicted)
        ref_keys = AnswerMetrics._extract_keywords(reference)

        if not pred_keys and not ref_keys:
            return 1.0
        if not pred_keys or not ref_keys:
            return 0.0

        common = set(pred_keys) & set(ref_keys)

        precision = len(common) / len(pred_keys)
        recall = len(common) / len(ref_keys)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

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
            "f1_score": AnswerMetrics.f1_score(predicted, reference)
        }
