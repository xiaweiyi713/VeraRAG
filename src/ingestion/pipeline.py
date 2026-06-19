"""Ingestion pipeline: load → chunk → index."""

import logging
from collections.abc import Mapping
from typing import Any

from ..retriever.base import BaseRetriever
from .chunker import Chunk, TextChunker
from .loader import DocumentLoader, RawDocument

logger = logging.getLogger("verarag.ingestion")

_DOCUMENT_FIELDS = {"id", "doc_id", "text", "content", "title", "metadata"}


class IngestionPipeline:
    """Full ingestion pipeline: load documents, chunk, and build indexes."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunk_strategy: str = "fixed",
        min_chunk_size: int = 50,
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy,
            min_chunk_size=min_chunk_size,
        )

    def ingest_file(self, file_path: str) -> list[Chunk]:
        docs = self.loader.load_file(file_path)
        return self.chunker.chunk_documents(docs)

    def ingest_directory(self, dir_path: str, recursive: bool = True) -> list[Chunk]:
        docs = self.loader.load_directory(dir_path, recursive=recursive)
        return self.chunker.chunk_documents(docs)

    def ingest_documents(self, docs: list[RawDocument]) -> list[Chunk]:
        return self.chunker.chunk_documents(docs)

    def build_retriever_index(
        self,
        chunks: list[Chunk],
        retriever_type: str = "bm25",
        retriever_config: dict[str, Any] | None = None,
    ) -> BaseRetriever:
        """Build a retriever index from chunks.

        Args:
            chunks: List of chunks to index
            retriever_type: 'bm25', 'dense', 'hybrid', or 'faiss'
            retriever_config: Optional config for retriever

        Returns:
            Initialized retriever with documents indexed
        """
        if not chunks:
            raise ValueError("Cannot build retriever index from zero chunks")

        retriever_type = retriever_type.lower().strip()
        index_docs = [c.to_index_doc() for c in chunks]

        if retriever_type == "bm25":
            from ..retriever.bm25 import BM25Retriever
            retriever: BaseRetriever = BM25Retriever(config=retriever_config)
        elif retriever_type == "dense":
            from ..retriever.dense import DenseRetriever
            model_name = (retriever_config or {}).get("model_name", "BAAI/bge-base-en-v1.5")
            retriever = DenseRetriever(model_name=model_name)
        elif retriever_type == "faiss":
            from ..retriever.dense import FAISSRetriever
            model_name = (retriever_config or {}).get("model_name", "BAAI/bge-base-en-v1.5")
            retriever = FAISSRetriever(model_name=model_name)
        elif retriever_type == "hybrid":
            from ..retriever.hybrid import HybridRetriever
            retriever = HybridRetriever(config=retriever_config)
        else:
            raise ValueError(
                "Unknown retriever type: "
                f"{retriever_type}. Expected one of: bm25, dense, faiss, hybrid"
            )

        logger.info(f"Indexing {len(index_docs)} chunks with {retriever_type}...")
        retriever.index_documents(index_docs)
        logger.info("Indexing complete.")

        return retriever

    def ingest_and_index(
        self,
        source_path: str,
        retriever_type: str = "bm25",
        retriever_config: dict[str, Any] | None = None,
        is_directory: bool = False,
    ) -> tuple[list[Chunk], BaseRetriever]:
        """Full pipeline: load → chunk → index.

        Returns:
            Tuple of (chunks, retriever)
        """
        if is_directory:
            chunks = self.ingest_directory(source_path)
        else:
            chunks = self.ingest_file(source_path)

        logger.info(f"Loaded and chunked into {len(chunks)} chunks from {source_path}")

        retriever = self.build_retriever_index(chunks, retriever_type, retriever_config)

        return chunks, retriever

    def ingest_and_index_from_docs(
        self,
        documents: list[Mapping[str, Any]],
        retriever_type: str = "bm25",
        retriever_config: dict[str, Any] | None = None,
    ) -> tuple[list[Chunk], BaseRetriever]:
        """Index from pre-parsed document list (id, text, title).

        Returns:
            Tuple of (chunks, retriever)
        """
        chunks: list[Chunk] = []
        seen_ids: set[str] = set()
        for i, doc in enumerate(documents):
            chunk = self._chunk_from_preparsed_document(doc, i)
            if chunk.doc_id in seen_ids:
                raise ValueError(f"Duplicate document id in pre-parsed documents: {chunk.doc_id}")
            seen_ids.add(chunk.doc_id)
            chunks.append(chunk)

        retriever = self.build_retriever_index(chunks, retriever_type, retriever_config)
        return chunks, retriever

    def _chunk_from_preparsed_document(
        self,
        document: Mapping[str, Any],
        index: int,
    ) -> Chunk:
        if not isinstance(document, Mapping):
            raise ValueError(f"Pre-parsed document {index} must be a mapping")

        doc_id = document.get("id", document.get("doc_id"))
        text = document.get("text", document.get("content"))
        title = document.get("title", "")
        metadata = document.get("metadata", {})

        if doc_id is None or str(doc_id).strip() == "":
            raise ValueError(f"Pre-parsed document {index} is missing a non-empty id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Pre-parsed document {index} is missing non-empty text")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Pre-parsed document {index} metadata must be a mapping")

        extra_metadata = {k: v for k, v in document.items() if k not in _DOCUMENT_FIELDS}
        merged_metadata = {**dict(metadata), **extra_metadata}
        stable_id = str(doc_id).strip()
        return Chunk(
            chunk_id=stable_id,
            doc_id=stable_id,
            text=text.strip(),
            title=str(title),
            metadata=merged_metadata,
        )
