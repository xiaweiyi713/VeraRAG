"""Public API for VeraRAG.

The historical internal package path is ``src.*``. New user-facing code should
prefer importing from ``verarag`` so the public API is not tied to repository
layout.
"""

from src.benchmark import VeraBenchEvaluator, VeraBenchLoader, load_verabench
from src.pipeline.verarag import VeraRAG, VeraRAGOutput, create_verarag

__all__ = [
    "VeraBenchEvaluator",
    "VeraBenchLoader",
    "VeraRAG",
    "VeraRAGOutput",
    "create_verarag",
    "load_verabench",
]
