"""Tests for document ingestion and chunking."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import Chunk, TextChunker
from src.ingestion.loader import DocumentLoader, RawDocument
from src.ingestion.pipeline import IngestionPipeline

# --- Fixtures ---

SAMPLE_JSONL = [
    {"doc_id": "D1", "title": "测试文档", "content": "这是第一段内容。" * 50, "source": "test"},
    {"doc_id": "D2", "title": "另一篇文档", "content": "这是第二段内容。" * 50, "source": "test"},
]

SAMPLE_TXT = "这是一个纯文本文档。\n\n第二段落。\n\n第三段落。"

SAMPLE_MD = """# 标题一

第一段内容。

## 标题二

第二段内容。

### 标题三

第三段内容。"""


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        jsonl_path = os.path.join(tmpdir, "test.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for doc in SAMPLE_JSONL:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        txt_path = os.path.join(tmpdir, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TXT)

        md_path = os.path.join(tmpdir, "test.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_MD)

        yield tmpdir


# --- Loader Tests ---

class TestDocumentLoader:
    def test_load_jsonl(self, temp_dir):
        loader = DocumentLoader()
        docs = loader.load_file(os.path.join(temp_dir, "test.jsonl"))
        assert len(docs) == 2
        assert docs[0].doc_id == "D1"
        assert docs[0].title == "测试文档"
        assert docs[1].doc_id == "D2"

    def test_load_txt(self, temp_dir):
        loader = DocumentLoader()
        docs = loader.load_file(os.path.join(temp_dir, "test.txt"))
        assert len(docs) == 1
        assert docs[0].doc_id == "test"
        assert "纯文本" in docs[0].content

    def test_load_markdown(self, temp_dir):
        loader = DocumentLoader()
        docs = loader.load_file(os.path.join(temp_dir, "test.md"))
        assert len(docs) >= 1
        # Markdown should be split by headings
        assert any("标题" in d.title for d in docs)

    def test_load_directory(self, temp_dir):
        loader = DocumentLoader()
        docs = loader.load_directory(temp_dir)
        assert len(docs) >= 4  # 2 JSONL + 1 TXT + 1+ MD

    def test_load_nonexistent(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.txt")

    def test_load_unsupported_format(self):
        loader = DocumentLoader()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                loader.load_file(f.name)
        finally:
            os.unlink(f.name)

    def test_load_verabench_corpus(self):
        loader = DocumentLoader()
        docs = loader.load_file("data/verabench/corpus.jsonl")
        assert len(docs) >= 33
        assert all(d.doc_id for d in docs)
        assert all(d.content for d in docs)


# --- Chunker Tests ---

class TestTextChunker:
    def test_fixed_chunk(self, temp_dir):
        loader = DocumentLoader()
        docs = loader.load_file(os.path.join(temp_dir, "test.jsonl"))
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="fixed")
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) > 2  # Should split long documents
        assert all(c.text for c in chunks)
        assert all(c.doc_id for c in chunks)

    def test_sentence_chunk(self):
        doc = RawDocument(doc_id="D1", title="T", content="第一句话。第二句话。第三句话。第四句话。第五句话。")
        chunker = TextChunker(chunk_size=30, chunk_overlap=10, strategy="sentence")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert all(c.text for c in chunks)

    def test_paragraph_chunk(self):
        text = "第一段内容。" * 10 + "\n\n" + "第二段内容。" * 10 + "\n\n" + "第三段内容。" * 10
        doc = RawDocument(doc_id="D1", title="T", content=text)
        chunker = TextChunker(chunk_size=200, strategy="paragraph")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_small_document_single_chunk(self):
        doc = RawDocument(doc_id="D1", title="T", content="短文本")
        chunker = TextChunker(chunk_size=512)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "短文本"

    def test_empty_document_no_chunks(self):
        doc = RawDocument(doc_id="D1", title="T", content="")
        chunker = TextChunker()
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_chunk_to_index_doc(self):
        chunk = Chunk(chunk_id="C1", doc_id="D1", text="text", title="Title", metadata={"key": "val"})
        d = chunk.to_index_doc()
        assert d["id"] == "C1"
        assert d["text"] == "text"
        assert d["doc_id"] == "D1"
        assert d["key"] == "val"

    def test_verabench_corpus_chunking(self):
        loader = DocumentLoader()
        docs = loader.load_file("data/verabench/corpus.jsonl")
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= 33
        assert all(c.text for c in chunks)


# --- Pipeline Tests ---

class TestIngestionPipeline:
    def test_ingest_file(self):
        pipeline = IngestionPipeline(chunk_size=512)
        chunks = pipeline.ingest_file("data/verabench/corpus.jsonl")
        assert len(chunks) >= 33

    def test_ingest_and_index_bm25(self):
        pipeline = IngestionPipeline(chunk_size=512)
        chunks, retriever = pipeline.ingest_and_index(
            "data/verabench/corpus.jsonl", retriever_type="bm25",
        )
        assert len(chunks) >= 33
        results = retriever.retrieve("欧盟AI法案", top_k=3)
        assert len(results) > 0
        assert any("AI" in r.title or "法案" in r.title for r in results)

    def test_bm25_chinese_retrieval(self):
        pipeline = IngestionPipeline(chunk_size=512)
        _chunks, retriever = pipeline.ingest_and_index(
            "data/verabench/corpus.jsonl", retriever_type="bm25",
        )
        # Test specific Chinese queries
        queries = [
            ("星辰科技营收", "星辰科技"),
            ("固态电池量产", "固态电池"),
            ("量子计算谷歌", "量子"),
        ]
        for query, expected_keyword in queries:
            results = retriever.retrieve(query, top_k=3)
            assert len(results) > 0, f"No results for query: {query}"
            assert any(expected_keyword in r.content or expected_keyword in r.title
                      for r in results), f"Expected '{expected_keyword}' in results for '{query}'"

    def test_ingest_directory(self, temp_dir):
        pipeline = IngestionPipeline(chunk_size=512)
        chunks = pipeline.ingest_directory(temp_dir)
        assert len(chunks) >= 2  # At least the JSONL docs
