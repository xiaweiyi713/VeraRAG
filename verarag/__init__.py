"""Public API for VeraRAG.

The historical internal package path is ``src.*``. New user-facing code should
prefer importing from ``verarag`` so the public API is not tied to repository
layout.
"""

import importlib.metadata as importlib_metadata

from src import __version__ as _SOURCE_VERSION
from src.benchmark import (
    ConflictPairExample,
    VeraBenchEvaluator,
    VeraBenchLoader,
    audit_external_conflict_set,
    audit_verabench_contamination,
    build_conflict_pair_examples,
    build_external_annotation_packet,
    compile_external_annotation_packet,
    load_verabench,
    summarize_conflict_pair_examples,
    write_conflict_pair_dataset,
)
from src.pipeline.verarag import VeraRAG, VeraRAGOutput, create_verarag


def _read_package_version(package_name: str = "verarag") -> str:
    """Return installed package metadata version, falling back for source checkouts."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return _SOURCE_VERSION


__version__ = _read_package_version()

__all__ = [
    "ConflictPairExample",
    "VeraBenchEvaluator",
    "VeraBenchLoader",
    "VeraRAG",
    "VeraRAGOutput",
    "__version__",
    "audit_external_conflict_set",
    "audit_verabench_contamination",
    "build_conflict_pair_examples",
    "build_external_annotation_packet",
    "compile_external_annotation_packet",
    "create_verarag",
    "load_verabench",
    "summarize_conflict_pair_examples",
    "write_conflict_pair_dataset",
]
