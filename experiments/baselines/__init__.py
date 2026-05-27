"""Baseline RAG implementations for comparison with VeraRAG."""

from .vanilla_rag import VanillaRAG
from .hybrid_rag import HybridRAG
from .self_rag import SelfRAG

__all__ = ["VanillaRAG", "HybridRAG", "SelfRAG"]
