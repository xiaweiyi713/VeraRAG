"""Ingestion pipeline: load → chunk → index."""

import logging
from typing import Any

from ..retriever.base import BaseRetriever
from .chunker import Chunk, TextChunker
from .loader import DocumentLoader, RawDocument

logger = logging.getLogger("verarag.ingestion")


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
            raise ValueError(f"Unknown retriever type: {retriever_type}")

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
        documents: list,
        retriever_type: str = "bm25",
        retriever_config: dict[str, Any] | None = None,
    ) -> tuple[list[Chunk], BaseRetriever]:
        """Index from pre-parsed document list (id, text, title).

        Returns:
            Tuple of (chunks, retriever)
        """
        from .chunker import Chunk

        chunks: list[Chunk] = []
        for doc in documents:
            chunk = Chunk(
                chunk_id=doc.get("id", ""),
                doc_id=doc.get("id", ""),
                text=doc.get("text", ""),
                title=doc.get("title", ""),
            )
            chunks.append(chunk)

        retriever = self.build_retriever_index(chunks, retriever_type, retriever_config)
        return chunks, retriever
