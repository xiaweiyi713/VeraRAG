"""Base Retriever class for VeraRAG."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    doc_id: str
    content: str
    title: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseRetriever(ABC):
    """Base class for all retrievers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for a query.

        Args:
            query: Query string
            top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of retrieval results
        """
        pass

    @abstractmethod
    def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Build index from documents.

        Args:
            documents: List of documents with 'id' and 'text' fields
        """
        pass

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query
            **kwargs: Additional arguments

        Returns:
            List of retrieval result lists
        """
        return [self.retrieve(q, top_k, **kwargs) for q in queries]

    def save_index(self, path: str) -> None:
        """Save index to disk."""
        pass

    def load_index(self, path: str) -> None:
        """Load index from disk."""
        pass
