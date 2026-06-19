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
        batch_size: int = 32,
        local_files_only: bool = False,
    ):
        super().__init__(config)
        config = config or {}
        self.model_name = str(config.get("model_name", model_name))
        self.device = str(config.get("device", device))
        self.batch_size = int(config.get("batch_size", batch_size))
        self.local_files_only = bool(
            config.get("local_files_only", local_files_only)
        )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.model: Any = None
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
                device=self.device,
                local_files_only=self.local_files_only,
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
        return self._as_2d_float_array(embeddings, expected_rows=len(texts))

    def _as_2d_float_array(
        self,
        embeddings: Any,
        expected_rows: int | None = None,
    ) -> np.ndarray:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("Dense embeddings must be a non-empty 2D array")
        if expected_rows is not None and array.shape[0] != expected_rows:
            raise ValueError(
                "Dense embedding row count does not match input texts: "
                f"{array.shape[0]} != {expected_rows}"
            )
        return array

    def _validate_index_state(self) -> None:
        if self.embeddings is None:
            return
        self.embeddings = self._as_2d_float_array(
            self.embeddings,
            expected_rows=len(self.doc_ids),
        )
        if len(self.doc_texts) != len(self.doc_ids):
            raise ValueError("Dense index doc_texts length does not match doc_ids")
        if len(self.doc_metadata) != len(self.doc_ids):
            raise ValueError("Dense index metadata length does not match doc_ids")

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Build dense index from documents.

        Args:
            documents: List of documents with 'id' and 'text' fields
        """
        self.doc_ids = []
        self.doc_texts = []
        self.doc_metadata = []

        if not documents:
            self.embeddings = None
            return

        for doc in documents:
            doc_id = str(doc.get('id', str(len(self.doc_ids))))
            text = str(doc.get('text', ''))
            title = str(doc.get('title', ''))

            # Combine title and text
            full_text = f"{title} {text}".strip() if title else text.strip()
            if not full_text:
                raise ValueError(f"Dense document {doc_id} has empty text")

            self.doc_ids.append(doc_id)
            self.doc_texts.append(full_text)
            self.doc_metadata.append({
                'title': title,
                **{k: v for k, v in doc.items() if k not in ['id', 'text', 'title']}
            })

        # Encode all documents
        self.embeddings = self._encode_texts(self.doc_texts)
        self._validate_index_state()

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
        query = self._validate_query(query)
        top_k = self._validate_top_k(top_k)
        if self.embeddings is None or not self.doc_ids or top_k <= 0 or not query.strip():
            return []
        self._validate_index_state()

        # Encode query
        query_embedding = self._encode_texts([query])[0]
        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match dense index: "
                f"{query_embedding.shape[0]} != {self.embeddings.shape[1]}"
            )

        # Compute cosine similarity
        scores = np.dot(self.embeddings, query_embedding)

        # Get L2 norms for normalization
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm_raw = np.linalg.norm(query_embedding)

        # Handle zero norms
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        query_norm = float(query_norm_raw) if query_norm_raw > 0 else 1.0

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
        if self.embeddings is None:
            raise ValueError("Cannot save dense index before indexing documents")
        self._validate_index_state()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'embeddings': self.embeddings,
            'doc_ids': self.doc_ids,
            'doc_texts': self.doc_texts,
            'doc_metadata': self.doc_metadata,
            'model_name': self.model_name,
            'local_files_only': self.local_files_only,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, path: str) -> None:
        """Load dense index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings = data['embeddings']
        self.doc_ids = [str(doc_id) for doc_id in data['doc_ids']]
        self.doc_texts = [str(text) for text in data['doc_texts']]
        self.doc_metadata = [dict(metadata) for metadata in data['doc_metadata']]
        self.model_name = str(data['model_name'])
        self.local_files_only = bool(data.get('local_files_only', self.local_files_only))
        self._validate_index_state()


class FAISSRetriever(DenseRetriever):
    """
    Dense retriever with FAISS indexing for efficient large-scale retrieval.
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.faiss_index: Any = None

    def _build_faiss_index(self) -> None:
        """Build FAISS index from embeddings."""
        import faiss

        if self.embeddings is None or not self.doc_ids:
            self.faiss_index = None
            return
        self._validate_index_state()
        assert self.embeddings is not None
        normalized_embeddings = self.embeddings.astype('float32', copy=True)
        dimension = normalized_embeddings.shape[1]
        # Use Inner Product (IP) for cosine similarity with normalized vectors
        self.faiss_index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(normalized_embeddings)
        self.faiss_index.add(normalized_embeddings)

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
        query = self._validate_query(query)
        top_k = self._validate_top_k(top_k)
        if self.faiss_index is None or top_k <= 0 or not query.strip():
            return []

        # Encode and normalize query
        query_embedding = self._encode_texts([query])[0]
        if self.embeddings is not None and query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match FAISS index: "
                f"{query_embedding.shape[0]} != {self.embeddings.shape[1]}"
            )
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

    def load_index(self, path: str) -> None:
        """Load dense index and rebuild the FAISS search structure."""
        super().load_index(path)
        self._build_faiss_index()
