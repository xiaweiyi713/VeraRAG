"""Base Retriever class for VeraRAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    doc_id: str
    content: str
    title: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Base class for all retrievers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def _validate_top_k(self, top_k: int) -> int:
        """Validate a retrieval limit shared by all retrievers."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        return top_k

    def _validate_query(self, query: str) -> str:
        """Validate a query string shared by all retrievers."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        return query

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        if not isinstance(queries, list):
            raise TypeError("queries must be a list of strings")
        for query in queries:
            self._validate_query(query)
        top_k = self._validate_top_k(top_k)
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
