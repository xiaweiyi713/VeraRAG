"""BM25 Sparse Retriever for VeraRAG."""

import pickle
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

from .base import BaseRetriever, RetrievalResult


class BM25Retriever(BaseRetriever):
    """
    BM25 sparse retriever using rank_bm25 implementation.

    BM25 is a bag-of-words retrieval function that ranks documents
    based on query terms appearing in each document.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        super().__init__(config)
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        self.index = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize with Chinese + English support."""
        text = text.lower()
        # Try jieba for Chinese word segmentation
        try:
            import jieba
            # jieba handles mixed Chinese/English text
            tokens = list(jieba.cut(text))
            # Filter whitespace and single chars (keep meaningful tokens)
            tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]
            return tokens
        except ImportError:
            # Fallback: character bigrams for CJK + word splitting for English
            tokens = re.findall(r'\w+', text)
            return tokens

    def _build_index(self) -> None:
        """Build BM25 index from corpus."""
        from rank_bm25 import BM25Okapi

        # Tokenize corpus
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]

        # Build BM25 index
        self.index = BM25Okapi(
            tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents with 'id' and 'text' fields
        """
        self.corpus = []
        self.doc_ids = []
        self.doc_metadata = []

        for doc in documents:
            doc_id = doc.get('id', str(len(self.doc_ids)))
            text = doc.get('text', '')
            title = doc.get('title', '')

            # Combine title and text
            full_text = f"{title} {text}" if title else text

            self.doc_ids.append(doc_id)
            self.corpus.append(full_text)
            self.doc_metadata.append({
                'title': title,
                **{k: v for k, v in doc.items() if k not in ['id', 'text', 'title']}
            })

        self._build_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using BM25.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of retrieval results sorted by score
        """
        if self.index is None:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get scores
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    doc_id=self.doc_ids[idx],
                    content=self.corpus[idx],
                    title=self.doc_metadata[idx].get('title', ''),
                    score=float(scores[idx]),
                    metadata=self.doc_metadata[idx]
                ))

        return results

    def save_index(self, path: str) -> None:
        """Save BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'corpus': self.corpus,
            'doc_ids': self.doc_ids,
            'doc_metadata': self.doc_metadata,
            'k1': self.k1,
            'b': self.b,
            'epsilon': self.epsilon
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, path: str) -> None:
        """Load BM25 index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.corpus = data['corpus']
        self.doc_ids = data['doc_ids']
        self.doc_metadata = data['doc_metadata']
        self.k1 = data['k1']
        self.b = data['b']
        self.epsilon = data['epsilon']

        self._build_index()


class SparseRetriever(BM25Retriever):
    """Alias for BM25Retriever for compatibility."""
    pass
