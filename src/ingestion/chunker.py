"""Document chunking strategies."""

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .loader import RawDocument


@dataclass
class Chunk:
    """A chunk of text from a document."""
    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_index_doc(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "title": self.title,
            "doc_id": self.doc_id,
            **self.metadata,
        }


class TextChunker:
    """Split documents into chunks with multiple strategies."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "fixed",
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        if not doc.content or len(doc.content.strip()) < self.min_chunk_size:
            # Document too small, return as single chunk
            if doc.content.strip():
                return [Chunk(
                    chunk_id=f"{doc.doc_id}_c0",
                    doc_id=doc.doc_id,
                    text=doc.content.strip(),
                    title=doc.title,
                    metadata={"source": doc.source, **doc.metadata},
                )]
            return []

        strategy_map = {
            "fixed": self._fixed_chunk,
            "sentence": self._sentence_chunk,
            "paragraph": self._paragraph_chunk,
            "heading": self._heading_chunk,
        }

        chunker = strategy_map.get(self.strategy, self._fixed_chunk)
        text_chunks = chunker(doc.content)

        chunks = []
        for i, text in enumerate(text_chunks):
            text = text.strip()
            if len(text) < self.min_chunk_size:
                if chunks:
                    chunks[-1].text += "\n" + text
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_c{i}",
                doc_id=doc.doc_id,
                text=text,
                title=doc.title,
                metadata={
                    "source": doc.source,
                    "chunk_index": i,
                    **doc.metadata,
                },
            ))

        return chunks

    def chunk_documents(self, docs: List[RawDocument]) -> List[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _fixed_chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sentence_chunk(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        return self._merge_sentences(sentences)

    def _paragraph_chunk(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return self._merge_paragraphs(paragraphs)

    def _heading_chunk(self, text: str) -> List[str]:
        sections = re.split(r'\n(?=#{1,4}\s)', text)
        return [s.strip() for s in sections if s.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        # Split on Chinese and English sentence endings
        parts = re.split(r'(?<=[。！？.!?])\s*', text)
        return [p.strip() for p in parts if p.strip()]

    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        chunks = []
        current = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Keep overlap
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > self.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sents
                current_len = overlap_len

            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _merge_paragraphs(self, paragraphs: List[str]) -> List[str]:
        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_len = para_len
            else:
                current.append(para)
                current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        return chunks
