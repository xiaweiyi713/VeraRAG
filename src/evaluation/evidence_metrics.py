"""Evidence Metrics for VeraRAG Evaluation."""

from typing import Dict, Any, List, Set

from ..utils.data_structures import Evidence


class EvidenceMetrics:
    """
    Metrics for evaluating evidence quality.

    Includes:
    - Evidence Precision: % of retrieved evidence that is relevant
    - Evidence Recall: % of relevant evidence that was retrieved
    - Evidence F1: Harmonic mean of precision and recall
    - Supporting Fact F1: For datasets like HotpotQA
    """

    @staticmethod
    def evidence_precision(
        retrieved_evidence: List[str],
        relevant_evidence: List[str]
    ) -> float:
        """
        Calculate evidence precision.

        Args:
            retrieved_evidence: IDs of retrieved evidence
            relevant_evidence: IDs of ground truth relevant evidence

        Returns:
            Precision score (0-1)
        """
        if not retrieved_evidence:
            return 0.0

        retrieved_set = set(retrieved_evidence)
        relevant_set = set(relevant_evidence)

        correct = len(retrieved_set & relevant_set)
        return correct / len(retrieved_set)

    @staticmethod
    def evidence_recall(
        retrieved_evidence: List[str],
        relevant_evidence: List[str]
    ) -> float:
        """
        Calculate evidence recall.

        Args:
            retrieved_evidence: IDs of retrieved evidence
            relevant_evidence: IDs of ground truth relevant evidence

        Returns:
            Recall score (0-1)
        """
        if not relevant_evidence:
            return 1.0

        retrieved_set = set(retrieved_evidence)
        relevant_set = set(relevant_evidence)

        correct = len(retrieved_set & relevant_set)
        return correct / len(relevant_set)

    @staticmethod
    def evidence_f1(
        retrieved_evidence: List[str],
        relevant_evidence: List[str]
    ) -> float:
        """
        Calculate evidence F1 score.

        Args:
            retrieved_evidence: IDs of retrieved evidence
            relevant_evidence: IDs of ground truth relevant evidence

        Returns:
            F1 score (0-1)
        """
        precision = EvidenceMetrics.evidence_precision(retrieved_evidence, relevant_evidence)
        recall = EvidenceMetrics.evidence_recall(retrieved_evidence, relevant_evidence)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def supporting_fact_precision(
        predicted_facts: List[str],
        gold_facts: List[str]
    ) -> float:
        """
        Calculate supporting fact precision (HotpotQA style).

        Args:
            predicted_facts: Predicted supporting fact IDs/sentences
            gold_facts: Gold supporting fact IDs/sentences

        Returns:
            Precision score (0-1)
        """
        if not predicted_facts:
            return 0.0

        pred_set = set(predicted_facts)
        gold_set = set(gold_facts)

        correct = len(pred_set & gold_set)
        return correct / len(pred_set)

    @staticmethod
    def supporting_fact_recall(
        predicted_facts: List[str],
        gold_facts: List[str]
    ) -> float:
        """
        Calculate supporting fact recall (HotpotQA style).

        Args:
            predicted_facts: Predicted supporting fact IDs/sentences
            gold_facts: Gold supporting fact IDs/sentences

        Returns:
            Recall score (0-1)
        """
        if not gold_facts:
            return 1.0

        pred_set = set(predicted_facts)
        gold_set = set(gold_facts)

        correct = len(pred_set & gold_set)
        return correct / len(gold_set)

    @staticmethod
    def supporting_fact_f1(
        predicted_facts: List[str],
        gold_facts: List[str]
    ) -> float:
        """
        Calculate supporting fact F1 score (HotpotQA style).

        Args:
            predicted_facts: Predicted supporting fact IDs/sentences
            gold_facts: Gold supporting fact IDs/sentences

        Returns:
            F1 score (0-1)
        """
        precision = EvidenceMetrics.supporting_fact_precision(predicted_facts, gold_facts)
        recall = EvidenceMetrics.supporting_fact_recall(predicted_facts, gold_facts)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def joint_em(
        answer_em: float,
        supporting_fact_em: float
    ) -> float:
        """
        Calculate joint EM (answer AND supporting facts both correct).

        Args:
            answer_em: Answer exact match score
            supporting_fact_em: Supporting fact exact match score

        Returns:
            Joint EM score (0-1)
        """
        return 1.0 if (answer_em == 1.0 and supporting_fact_em == 1.0) else 0.0

    @staticmethod
    def citation_precision(
        answer: str,
        evidence_map: Dict[str, Evidence],
        valid_citations: Set[str]
    ) -> float:
        """
        Calculate citation precision.

        Args:
            answer: Generated answer with citations like [E1]
            evidence_map: Mapping from evidence IDs to Evidence objects
            valid_citations: Set of valid citation IDs

        Returns:
            Precision score (0-1)
        """
        # Extract citations from answer
        import re
        citations = re.findall(r'\[([E\d]+)\]', answer)

        if not citations:
            return 0.0

        valid = sum(1 for c in citations if c in valid_citations)
        return valid / len(citations)

    @staticmethod
    def citation_recall(
        answer: str,
        required_citations: Set[str]
    ) -> float:
        """
        Calculate citation recall.

        Args:
            answer: Generated answer with citations
            required_citations: Set of citation IDs that should be present

        Returns:
            Recall score (0-1)
        """
        if not required_citations:
            return 1.0

        import re
        citations = set(re.findall(r'\[([E\d]+)\]', answer))

        cited = len(citations & required_citations)
        return cited / len(required_citations)
