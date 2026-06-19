"""VeraBench: Benchmark for Verifiable Agentic RAG Evaluation."""

from .conflict_pairs import (
    ConflictPairExample,
    build_conflict_pair_examples,
    summarize_conflict_pair_examples,
    write_conflict_pair_dataset,
)
from .contamination import audit_verabench_contamination
from .evaluator import VeraBenchEvaluator
from .external_annotations import (
    audit_external_conflict_set,
    build_external_annotation_packet,
    compile_external_annotation_packet,
)
from .loader import VeraBenchLoader, load_verabench

__all__ = [
    "ConflictPairExample",
    "VeraBenchEvaluator",
    "VeraBenchLoader",
    "audit_external_conflict_set",
    "audit_verabench_contamination",
    "build_conflict_pair_examples",
    "build_external_annotation_packet",
    "compile_external_annotation_packet",
    "load_verabench",
    "summarize_conflict_pair_examples",
    "write_conflict_pair_dataset",
]
