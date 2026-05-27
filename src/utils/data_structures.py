"""
VeraRAG Core Data Structures

This module defines the fundamental data structures used across the VeraRAG system,
including Evidence, Claim, Conflict Graph, and Task Analysis results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ClaimType(Enum):
    """Types of claims extracted from evidence."""
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    UNCERTAINTY = "uncertainty"
    DEFINITIONAL = "definitional"


class TaskType(Enum):
    """Types of knowledge tasks."""
    MULTI_HOP_QA = "multi-hop_qa"
    FACT_VERIFICATION = "fact_verification"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TEMPORAL_REASONING = "temporal_reasoning"
    FINANCIAL_REASONING = "financial_reasoning"
    SCIENTIFIC_REVIEW = "scientific_review"


class Complexity(Enum):
    """Task complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConflictType(Enum):
    """Types of conflicts between evidence/claims."""
    SUPPORT = "support"
    REFUTE = "refute"
    PARTIAL_SUPPORT = "partial_support"
    NUMERIC_CONFLICT = "numeric_conflict"
    TEMPORAL_CONFLICT = "temporal_conflict"
    ENTITY_MISMATCH = "entity_mismatch"
    SOURCE_DISAGREEMENT = "source_disagreement"
    DEFINITIONAL_CONFLICT = "definitional_conflict"
    UNRELATED = "unrelated"


class VerificationStatus(Enum):
    """Verification status for claims."""
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_INFO = "not_enough_info"


@dataclass
class Claim:
    """A single atomic claim extracted from evidence."""
    claim_id: str
    claim: str
    claim_type: ClaimType
    entities: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    time_expressions: list[str] = field(default_factory=list)
    source_span: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "claim_type": self.claim_type.value,
            "entities": self.entities,
            "numbers": self.numbers,
            "time_expressions": self.time_expressions,
            "source_span": self.source_span,
            "confidence": self.confidence
        }


@dataclass
class Evidence:
    """A normalized evidence unit with metadata and claims."""
    evidence_id: str
    source: str  # paper, web, report, wiki, legal_case, etc.
    title: str
    text_span: str
    date: str | None = None
    author: str | None = None
    url: str | None = None
    entities: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    credibility_score: float = 0.8
    recency_score: float = 0.8
    relevance_score: float = 0.8

    @property
    def combined_score(self) -> float:
        """Combined evidence quality score."""
        return (
            self.credibility_score * 0.4 +
            self.recency_score * 0.3 +
            self.relevance_score * 0.3
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "title": self.title,
            "date": self.date,
            "author": self.author,
            "url": self.url,
            "text_span": self.text_span,
            "entities": self.entities,
            "claims": [c.to_dict() for c in self.claims],
            "credibility_score": self.credibility_score,
            "recency_score": self.recency_score,
            "relevance_score": self.relevance_score,
            "combined_score": self.combined_score
        }


@dataclass
class ConflictEdge:
    """An edge in the evidence conflict graph."""
    source_id: str
    target_id: str
    conflict_type: ConflictType
    confidence: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "conflict_type": self.conflict_type.value,
            "confidence": self.confidence,
            "rationale": self.rationale
        }


@dataclass
class ConflictGraphNode:
    """A node in the evidence conflict graph (can be claim or evidence)."""
    node_id: str
    content: str
    node_type: Literal["claim", "evidence"]
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type,
            "evidence_ids": self.evidence_ids
        }


class EvidenceConflictGraph:
    """
    Evidence Conflict Graph for modeling relationships between evidence and claims.

    This graph explicitly models:
    - Support relationships
    - Refutation relationships
    - Temporal conflicts
    - Numerical conflicts
    - Entity mismatches
    - Source disagreements
    """

    def __init__(self):
        self.nodes: dict[str, ConflictGraphNode] = {}
        self.edges: list[ConflictEdge] = []

    def add_node(self, node: ConflictGraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: ConflictEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def get_conflicts(self) -> list[ConflictEdge]:
        """Get all conflict-type edges (non-support edges)."""
        return [
            e for e in self.edges
            if e.conflict_type in {
                ConflictType.REFUTE,
                ConflictType.NUMERIC_CONFLICT,
                ConflictType.TEMPORAL_CONFLICT,
                ConflictType.ENTITY_MISMATCH,
                ConflictType.SOURCE_DISAGREEMENT,
                ConflictType.DEFINITIONAL_CONFLICT
            }
        ]

    def get_supports(self) -> list[ConflictEdge]:
        """Get all support-type edges."""
        return [
            e for e in self.edges
            if e.conflict_type == ConflictType.SUPPORT
        ]

    def get_conflict_score(self) -> float:
        """
        Calculate overall conflict score for the graph.

        Higher score indicates more conflicts.
        """
        if not self.edges:
            return 0.0
        conflict_count = len(self.get_conflicts())
        total_count = len(self.edges)
        return conflict_count / total_count if total_count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "conflict_score": self.get_conflict_score(),
            "num_conflicts": len(self.get_conflicts()),
            "num_supports": len(self.get_supports())
        }


@dataclass
class SubQuestion:
    """A sub-question from decomposition."""
    id: str
    question: str
    required_evidence_type: str = "general"
    dependency_ids: list[str] = field(default_factory=list)
    requires_counter_evidence: bool = False
    status: Literal["pending", "in_progress", "resolved", "unresolvable"] = "pending"
    coverage_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "required_evidence_type": self.required_evidence_type,
            "dependency_ids": self.dependency_ids,
            "requires_counter_evidence": self.requires_counter_evidence,
            "status": self.status,
            "coverage_score": self.coverage_score
        }


@dataclass
class TaskAnalysis:
    """Result of task analysis."""
    task_type: TaskType
    complexity: Complexity
    requires_retrieval: bool = True
    requires_conflict_check: bool = False
    requires_numerical_reasoning: bool = False
    requires_temporal_reasoning: bool = False
    estimated_hops: int = 1
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "complexity": self.complexity.value,
            "requires_retrieval": self.requires_retrieval,
            "requires_conflict_check": self.requires_conflict_check,
            "requires_numerical_reasoning": self.requires_numerical_reasoning,
            "requires_temporal_reasoning": self.requires_temporal_reasoning,
            "estimated_hops": self.estimated_hops,
            "keywords": self.keywords
        }


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step: int
    description: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "description": self.description,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence
        }


@dataclass
class AnswerClaim:
    """A claim in the final answer with verification info."""
    claim: str
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.8
    verification_status: VerificationStatus = VerificationStatus.NOT_ENOUGH_INFO

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "supporting_evidence": self.supporting_evidence,
            "conflicting_evidence": self.conflicting_evidence,
            "confidence": self.confidence,
            "verification_status": self.verification_status.value
        }


@dataclass
class UncertaintyBreakdown:
    """Breakdown of uncertainty sources."""
    retrieval_uncertainty: float = 0.0
    evidence_conflict: float = 0.0
    reasoning_gap: float = 0.0
    source_reliability: float = 0.0
    verification_uncertainty: float = 0.0

    @property
    def overall(self) -> float:
        """Calculate overall uncertainty score."""
        return (
            self.retrieval_uncertainty * 0.25 +
            self.evidence_conflict * 0.30 +
            self.reasoning_gap * 0.20 +
            self.source_reliability * 0.15 +
            self.verification_uncertainty * 0.10
        )

    def is_acceptable(self, threshold: float = 0.3) -> bool:
        """Check if uncertainty is acceptable for answering."""
        return self.overall < threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_uncertainty": self.retrieval_uncertainty,
            "evidence_conflict": self.evidence_conflict,
            "reasoning_gap": self.reasoning_gap,
            "source_reliability": self.source_reliability,
            "verification_uncertainty": self.verification_uncertainty,
            "overall_uncertainty": self.overall
        }


@dataclass
class VerificationReport:
    """Report from the verifier agent."""
    claim_verifications: list[dict[str, Any]] = field(default_factory=list)
    overall_status: VerificationStatus = VerificationStatus.NOT_ENOUGH_INFO
    issues: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence_for: list[str] = field(default_factory=list)
    overconfident_claims: list[str] = field(default_factory=list)
    ignored_conflicts: list[dict[str, Any]] = field(default_factory=list)

    def has_critical_issues(self) -> bool:
        """Check if there are critical issues that require repair."""
        return (
            len(self.issues) > 0 or
            len(self.missing_evidence_for) > 0 or
            len(self.ignored_conflicts) > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_verifications": self.claim_verifications,
            "overall_status": self.overall_status.value,
            "issues": self.issues,
            "missing_evidence_for": self.missing_evidence_for,
            "overconfident_claims": self.overconfident_claims,
            "ignored_conflicts": self.ignored_conflicts,
            "has_critical_issues": self.has_critical_issues()
        }


@dataclass
class VeraRAGOutput:
    """
    Final output of the VeraRAG system.

    Contains answer, evidence, conflict report, and uncertainty information.
    """
    question: str
    answer: str
    answer_claims: list[AnswerClaim]
    evidence: list[Evidence]
    reasoning_chain: list[ReasoningStep]
    conflict_report: dict[str, Any]
    verification_report: VerificationReport | None = None
    confidence: float = 0.0
    uncertainty: UncertaintyBreakdown = field(default_factory=UncertaintyBreakdown)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "answer_claims": [c.to_dict() for c in self.answer_claims],
            "evidence": [e.to_dict() for e in self.evidence],
            "reasoning_chain": [r.to_dict() for r in self.reasoning_chain],
            "conflict_report": self.conflict_report,
            "verification_report": self.verification_report.to_dict() if self.verification_report else None,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty.to_dict(),
            "metadata": self.metadata
        }
