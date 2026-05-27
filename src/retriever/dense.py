"""Dense Retriever using sentence embeddings for VeraRAG."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseRetriever, RetrievalResult


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using sentence transformer embeddings.

    Uses semantic similarity between query and document embeddings
    for retrieval.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32
    ):
        super().__init__(config)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.embeddings: np.ndarray | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_metadata: list[dict[str, Any]] = []

    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Build dense index from documents.

        Args:
            documents: List of documents with 'id' and 'text' fields
        """
        self.doc_ids = []
        self.doc_texts = []
        self.doc_metadata = []

        for doc in documents:
            doc_id = doc.get('id', str(len(self.doc_ids)))
            text = doc.get('text', '')
            title = doc.get('title', '')

            # Combine title and text
            full_text = f"{title} {text}" if title else text

            self.doc_ids.append(doc_id)
            self.doc_texts.append(full_text)
            self.doc_metadata.append({
                'title': title,
                **{k: v for k, v in doc.items() if k not in ['id', 'text', 'title']}
            })

        # Encode all documents
        self.embeddings = self._encode_texts(self.doc_texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> list[RetrievalResult]:
        """
        Retrieve documents using dense similarity.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of retrieval results sorted by similarity score
        """
        if self.embeddings is None:
            return []

        # Encode query
        query_embedding = self._encode_texts([query])[0]

        # Compute cosine similarity
        scores = np.dot(self.embeddings, query_embedding)

        # Get L2 norms for normalization
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)

        # Handle zero norms
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        query_norm = query_norm if query_norm > 0 else 1

        # Cosine similarity
        scores = scores / (doc_norms * query_norm)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                doc_id=self.doc_ids[idx],
                content=self.doc_texts[idx],
                title=self.doc_metadata[idx].get('title', ''),
                score=float(scores[idx]),
                metadata=self.doc_metadata[idx]
            ))

        return results

    def save_index(self, path: str) -> None:
        """Save dense index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'embeddings': self.embeddings,
            'doc_ids': self.doc_ids,
            'doc_texts': self.doc_texts,
            'doc_metadata': self.doc_metadata,
            'model_name': self.model_name
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, path: str) -> None:
        """Load dense index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings = data['embeddings']
        self.doc_ids = data['doc_ids']
        self.doc_texts = data['doc_texts']
        self.doc_metadata = data['doc_metadata']
        self.model_name = data['model_name']


class FAISSRetriever(DenseRetriever):
    """
    Dense retriever with FAISS indexing for efficient large-scale retrieval.
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.faiss_index = None

    def _build_faiss_index(self) -> None:
        """Build FAISS index from embeddings."""
        import faiss

        dimension = self.embeddings.shape[1]
        # Use Inner Product (IP) for cosine similarity with normalized vectors
        self.faiss_index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings.astype('float32'))

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """Build FAISS index from documents."""
        super().index_documents(documents)
        self._build_faiss_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> list[RetrievalResult]:
        """Retrieve using FAISS index."""
        if self.faiss_index is None:
            return []

        # Encode and normalize query
        query_embedding = self._encode_texts([query])[0]
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        import faiss
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):  # noqa: B905
            if idx >= 0 and idx < len(self.doc_ids):
                results.append(RetrievalResult(
                    doc_id=self.doc_ids[idx],
                    content=self.doc_texts[idx],
                    title=self.doc_metadata[idx].get('title', ''),
                    score=float(score),
                    metadata=self.doc_metadata[idx]
                ))

        return results
