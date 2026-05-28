"""Base Retriever class for VeraRAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    doc_id: str
    content: str
    title: str = ""
    score: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseRetriever(ABC):
    """Base class for all retrievers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> list[RetrievalResult]:
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
        documents: list[dict[str, Any]]
    ) -> None:
        """
        Build index from documents.

        Args:
            documents: List of documents with 'id' and 'text' fields
        """
        pass

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int = 10,
        **kwargs
    ) -> list[list[RetrievalResult]]:
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support save_index"
        )

    def load_index(self, path: str) -> None:
        """Load index from disk."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support load_index"
        )
