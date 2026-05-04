"""Conflict Metrics for VeraRAG Evaluation."""

from typing import Dict, Any, List, Set, Tuple, Optional

from ..utils.data_structures import EvidenceConflictGraph, ConflictType


class ConflictMetrics:
    """
    Metrics for evaluating conflict detection and resolution.

    Includes:
    - Conflict Detection F1: How well conflicts are detected
    - Conflict Type Accuracy: Accuracy of conflict type classification
    - Conflict Resolution Accuracy: How well conflicts are resolved
    - False Conflict Rate: Rate of incorrect conflict detection
    """

    @staticmethod
    def conflict_detection_f1(
        predicted_conflicts: List[Tuple[str, str]],
        gold_conflicts: List[Tuple[str, str]]
    ) -> float:
        """
        Calculate F1 score for conflict detection.

        Args:
            predicted_conflicts: List of (source_id, target_id) tuples for predicted conflicts
            gold_conflicts: List of (source_id, target_id) tuples for gold conflicts

        Returns:
            F1 score (0-1)
        """
        pred_set = set(predicted_conflicts)
        gold_set = set(gold_conflicts)

        if not pred_set and not gold_set:
            return 1.0
        if not pred_set or not gold_set:
            return 0.0

        # Remove direction (treat A->B and B->A as same conflict)
        pred_undirected = {tuple(sorted(pair)) for pair in pred_set}
        gold_undirected = {tuple(sorted(pair)) for pair in gold_set}

        true_positives = len(pred_undirected & gold_undirected)
        false_positives = len(pred_undirected - gold_undirected)
        false_negatives = len(gold_undirected - pred_undirected)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def conflict_type_accuracy(
        predicted_types: List[Tuple[Tuple[str, str], ConflictType]],
        gold_types: List[Tuple[Tuple[str, str], ConflictType]]
    ) -> float:
        """
        Calculate accuracy of conflict type classification.

        Args:
            predicted_types: List of ((source_id, target_id), conflict_type) tuples
            gold_types: List of ((source_id, target_id), conflict_type) tuples

        Returns:
            Accuracy score (0-1)
        """
        if not gold_types:
            return 1.0

        # Create mapping
        pred_map = {tuple(sorted(pair)): ct for (pair, ct) in predicted_types}
        gold_map = {tuple(sorted(pair)): ct for (pair, ct) in gold_types}

        correct = 0
        total = 0

        for pair, gold_type in gold_map.items():
            if pair in pred_map:
                total += 1
                if pred_map[pair] == gold_type:
                    correct += 1

        return correct / total if total > 0 else 0.0

    @staticmethod
    def conflict_resolution_accuracy(
        resolution_decisions: List[Dict[str, Any]],
        gold_resolutions: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate accuracy of conflict resolution decisions.

        Args:
            resolution_decisions: Predicted resolution decisions
            gold_resolutions: Gold standard resolution decisions

        Returns:
            Accuracy score (0-1)
        """
        if not gold_resolutions:
            return 1.0

        # For each gold resolution, check if predicted resolution matches
        correct = 0

        for gold in gold_resolutions:
            conflict_pair = gold.get("conflict_pair")
            gold_resolution = gold.get("resolution")

            # Find matching predicted resolution
            for pred in resolution_decisions:
                if pred.get("conflict_pair") == conflict_pair:
                    if pred.get("resolution") == gold_resolution:
                        correct += 1
                    break

        return correct / len(gold_resolutions)

    @staticmethod
    def false_conflict_rate(
        predicted_conflicts: List[Tuple[str, str]],
        gold_conflicts: List[Tuple[str, str]]
    ) -> float:
        """
        Calculate rate of false conflict detections.

        Args:
            predicted_conflicts: List of predicted conflict pairs
            gold_conflicts: List of gold conflict pairs

        Returns:
            False conflict rate (0-1)
        """
        if not predicted_conflicts:
            return 0.0

        pred_set = {tuple(sorted(pair)) for pair in predicted_conflicts}
        gold_set = {tuple(sorted(pair)) for pair in gold_conflicts}

        false_positives = len(pred_set - gold_set)
        return false_positives / len(pred_set)

    @staticmethod
    def compute_all(
        predicted_conflicts: List[Tuple[str, str]],
        gold_conflicts: List[Tuple[str, str]],
        predicted_types: Optional[List[Tuple[Tuple[str, str], ConflictType]]] = None,
        gold_types: Optional[List[Tuple[Tuple[str, str], ConflictType]]] = None
    ) -> Dict[str, float]:
        """
        Compute all conflict metrics.

        Args:
            predicted_conflicts: Predicted conflict pairs
            gold_conflicts: Gold conflict pairs
            predicted_types: Optional predicted conflict types
            gold_types: Optional gold conflict types

        Returns:
            Dictionary with metric values
        """
        metrics = {
            "conflict_detection_f1": ConflictMetrics.conflict_detection_f1(
                predicted_conflicts, gold_conflicts
            ),
            "false_conflict_rate": ConflictMetrics.false_conflict_rate(
                predicted_conflicts, gold_conflicts
            )
        }

        if predicted_types and gold_types:
            metrics["conflict_type_accuracy"] = ConflictMetrics.conflict_type_accuracy(
                predicted_types, gold_types
            )

        return metrics
