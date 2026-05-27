"""Hallucination Metrics for VeraRAG Evaluation."""

import re


class HallucinationMetrics:
    """
    Metrics for evaluating hallucination in generated answers.

    Includes:
    - Unsupported Claim Rate: % of claims without evidence support
    - Entity Hallucination Rate: % of entities not in evidence
    - Numerical Hallucination Rate: % of numbers not in evidence
    - Overclaiming Rate: % of answers that are too confident
    """

    @staticmethod
    def unsupported_claim_rate(
        answer_claims: list[str],
        supported_claims: set[str]
    ) -> float:
        """
        Calculate rate of unsupported claims.

        Args:
            answer_claims: All claims made in the answer
            supported_claims: Set of claims that have evidence support

        Returns:
            Unsupported claim rate (0-1)
        """
        if not answer_claims:
            return 0.0

        unsupported = sum(1 for claim in answer_claims if claim not in supported_claims)
        return unsupported / len(answer_claims)

    @staticmethod
    def entity_hallucination_rate(
        answer_entities: set[str],
        evidence_entities: set[str]
    ) -> float:
        """
        Calculate rate of hallucinated entities.

        Args:
            answer_entities: Entities mentioned in the answer
            evidence_entities: Entities found in the evidence

        Returns:
            Entity hallucination rate (0-1)
        """
        if not answer_entities:
            return 0.0

        hallucinated = len(answer_entities - evidence_entities)
        return hallucinated / len(answer_entities)

    @staticmethod
    def numerical_hallucination_rate(
        answer_numbers: list[float],
        evidence_numbers: set[float],
        tolerance: float = 0.01
    ) -> float:
        """
        Calculate rate of hallucinated numerical values.

        Args:
            answer_numbers: Numbers mentioned in the answer
            evidence_numbers: Numbers found in the evidence
            tolerance: Tolerance for numerical matching

        Returns:
            Numerical hallucination rate (0-1)
        """
        if not answer_numbers:
            return 0.0

        hallucinated = 0
        for num in answer_numbers:
            # Check if number exists in evidence (with tolerance)
            found = any(
                abs(num - ev_num) <= tolerance * max(abs(num), abs(ev_num))
                for ev_num in evidence_numbers
            )
            if not found:
                hallucinated += 1

        return hallucinated / len(answer_numbers)

    @staticmethod
    def overclaiming_rate(
        confidences: list[float],
        correct: list[bool],
        confidence_threshold: float = 0.8
    ) -> float:
        """
        Calculate rate of overconfident incorrect answers.

        Args:
            confidences: Confidence scores for answers
            correct: Whether each answer was correct
            confidence_threshold: Threshold for "high confidence"

        Returns:
            Overclaiming rate (0-1)
        """
        high_confidence_count = 0
        overconfident_wrong = 0

        for conf, corr in zip(confidences, correct):  # noqa: B905
            if conf >= confidence_threshold:
                high_confidence_count += 1
                if not corr:
                    overconfident_wrong += 1

        if high_confidence_count == 0:
            return 0.0

        return overconfident_wrong / high_confidence_count

    @staticmethod
    def extract_entities_from_text(text: str) -> set[str]:
        """
        Extract named entities from text (simple heuristic).

        Args:
            text: Input text

        Returns:
            Set of entity strings
        """
        # Simple capitalization heuristic
        entities = set(re.findall(r'\b[A-Z][a-z]+\b', text))

        # Filter out common words
        stopwords = {'The', 'This', 'That', 'These', 'Those', 'It', 'Its'}
        entities -= stopwords

        return entities

    @staticmethod
    def extract_numbers_from_text(text: str) -> list[float]:
        """
        Extract numerical values from text.

        Args:
            text: Input text

        Returns:
            List of numerical values
        """
        # Match percentages first (to avoid overlapping with regular numbers)
        pattern = r'(\d+(?:,\d{3})*(?:\.\d+)?)%'

        numbers = []
        remaining_text = text

        # First extract percentages
        for match in re.finditer(pattern, text):
            try:
                clean = match.group(1).replace(',', '')
                num = float(clean) / 100
                numbers.append(num)
                # Remove matched portion from remaining text
                remaining_text = remaining_text.replace(match.group(0), '', 1)
            except ValueError:
                pass

        # Then extract regular numbers from remaining text
        for match in re.finditer(r'\d+(?:,\d{3})*(?:\.\d+)?', remaining_text):
            try:
                clean = match.group(0).replace(',', '')
                num = float(clean)
                numbers.append(num)
            except ValueError:
                pass

        return numbers

    @staticmethod
    def extract_claims_from_text(text: str) -> list[str]:
        """
        Extract claims from text (sentence-level).

        Args:
            text: Input text

        Returns:
            List of claim strings
        """
        # Split by sentence delimiters
        sentences = re.split(r'[.!?]+', text)

        # Filter and clean
        claims = [
            s.strip().lower()
            for s in sentences
            if len(s.strip()) > 10  # Minimum claim length
        ]

        return claims

    @staticmethod
    def compute_all(
        answer: str,
        evidence_texts: list[str],
        confidences: list[float],
        correct: list[bool]
    ) -> dict[str, float]:
        """
        Compute all hallucination metrics.

        Args:
            answer: Generated answer
            evidence_texts: Evidence texts to compare against
            confidences: Confidence scores
            correct: Correctness indicators

        Returns:
            Dictionary with metric values
        """
        # Extract entities and numbers from answer
        answer_entities = HallucinationMetrics.extract_entities_from_text(answer)
        answer_numbers = HallucinationMetrics.extract_numbers_from_text(answer)

        # Extract from evidence
        evidence_entities = set()
        evidence_numbers = set()
        for ev_text in evidence_texts:
            evidence_entities.update(HallucinationMetrics.extract_entities_from_text(ev_text))
            evidence_numbers.update(HallucinationMetrics.extract_numbers_from_text(ev_text))

        # Compute metrics
        return {
            "entity_hallucination_rate": HallucinationMetrics.entity_hallucination_rate(
                answer_entities, evidence_entities
            ),
            "numerical_hallucination_rate": HallucinationMetrics.numerical_hallucination_rate(
                answer_numbers, evidence_numbers
            ),
            "overclaiming_rate": HallucinationMetrics.overclaiming_rate(
                confidences, correct
            )
        }
