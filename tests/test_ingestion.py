"""Tests for document ingestion and chunking."""

import builtins
import json
import os
import sys
import tempfile
import types
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

    def test_load_json_single_document_preserves_metadata(self, tmp_path):
        path = tmp_path / "single.json"
        path.write_text(
            json.dumps(
                {
                    "id": 123,
                    "title": "JSON Title",
                    "text": "JSON content",
                    "source": "fixture",
                    "date": "2026-06-18",
                }
            ),
            encoding="utf-8",
        )

        docs = DocumentLoader().load_file(str(path))

        assert docs == [
            RawDocument(
                doc_id="123",
                title="JSON Title",
                content="JSON content",
                source="fixture",
                metadata={"date": "2026-06-18"},
            )
        ]

    def test_load_json_array_uses_fallback_ids_and_text_alias(self, tmp_path):
        path = tmp_path / "items.json"
        path.write_text(
            json.dumps(
                [
                    {"text": "first", "category": "a"},
                    {"content": "second", "title": "Second"},
                ]
            ),
            encoding="utf-8",
        )

        docs = DocumentLoader().load_file(str(path))

        assert [doc.doc_id for doc in docs] == ["items_0", "items_1"]
        assert [doc.content for doc in docs] == ["first", "second"]
        assert docs[0].metadata == {"category": "a"}

    def test_jsonl_reports_line_number_for_malformed_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"content": "ok"}\n{"content": \n', encoding="utf-8")

        with pytest.raises(ValueError, match=r"bad\.jsonl at line 2"):
            DocumentLoader().load_file(str(path))

    def test_jsonl_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blank_lines.jsonl"
        path.write_text('\n{"content": "ok"}\n   \n', encoding="utf-8")

        docs = DocumentLoader().load_file(str(path))

        assert len(docs) == 1
        assert docs[0].doc_id == "blank_lines_1"

    def test_json_rejects_non_object_root(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('"not a document"', encoding="utf-8")

        with pytest.raises(ValueError, match="object or array of objects"):
            DocumentLoader().load_file(str(path))

    def test_json_array_rejects_non_object_items(self, tmp_path):
        path = tmp_path / "bad_items.json"
        path.write_text(json.dumps([{"content": "ok"}, "bad"]), encoding="utf-8")

        with pytest.raises(ValueError, match=r"bad_items\.json item 2"):
            DocumentLoader().load_file(str(path))

    def test_json_content_must_be_string(self, tmp_path):
        path = tmp_path / "bad_content.json"
        path.write_text(json.dumps({"content": ["not", "text"]}), encoding="utf-8")

        with pytest.raises(ValueError, match="content must be a string"):
            DocumentLoader().load_file(str(path))

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

    def test_markdown_without_headings_returns_single_document(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("plain markdown\nwithout headings", encoding="utf-8")

        docs = DocumentLoader().load_file(str(path))

        assert docs == [
            RawDocument(
                doc_id="notes",
                title="notes",
                content="plain markdown\nwithout headings",
                source="file",
                metadata={"file_path": str(path), "format": "md"},
            )
        ]

    def test_markdown_preface_uses_fallback_section_title(self, tmp_path):
        path = tmp_path / "mixed.md"
        path.write_text("preface\n\n# Heading\nbody", encoding="utf-8")

        docs = DocumentLoader().load_file(str(path))

        assert [doc.title for doc in docs] == ["Section 1", "Heading"]
        assert [doc.metadata["section"] for doc in docs] == [0, 1]

    def test_markdown_skips_empty_split_sections(self, tmp_path):
        path = tmp_path / "empty_section.md"
        path.write_text("\n# First\nbody\n\n# Second\nmore", encoding="utf-8")

        docs = DocumentLoader().load_file(str(path))

        assert [doc.title for doc in docs] == ["First", "Second"]

    def test_load_pdf_uses_fitz_pages_and_closes_document(self, tmp_path, monkeypatch):
        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF fake")
        closed = []

        class FakePage:
            def __init__(self, text):
                self.text = text

            def get_text(self):
                return self.text

        class FakeDoc:
            def __init__(self):
                self.pages = [FakePage("first page"), FakePage("   "), FakePage("third page")]

            def __len__(self):
                return len(self.pages)

            def __getitem__(self, index):
                return self.pages[index]

            def close(self):
                closed.append(True)

        monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda _: FakeDoc()))

        docs = DocumentLoader().load_file(str(path))

        assert closed == [True]
        assert [doc.doc_id for doc in docs] == ["sample_p0", "sample_p2"]
        assert docs[0].source == "file"
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 3

    def test_empty_pdf_returns_file_source_fallback_and_closes(self, tmp_path, monkeypatch):
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"%PDF fake")
        closed = []

        class FakeDoc:
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError(index)

            def close(self):
                closed.append(True)

        monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda _: FakeDoc()))

        docs = DocumentLoader().load_file(str(path))

        assert closed == [True]
        assert docs == [
            RawDocument(
                doc_id="empty",
                title="empty",
                content="",
                source="file",
                metadata={"file_path": str(path), "format": "pdf"},
            )
        ]

    def test_load_pdf_closes_document_when_page_extraction_fails(self, tmp_path, monkeypatch):
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF fake")
        closed = []

        class FakePage:
            def get_text(self):
                raise RuntimeError("page failed")

        class FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return FakePage()

            def close(self):
                closed.append(True)

        monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda _: FakeDoc()))

        with pytest.raises(RuntimeError, match="page failed"):
            DocumentLoader().load_file(str(path))

        assert closed == [True]

    def test_load_pdf_reports_missing_pymupdf(self, tmp_path, monkeypatch):
        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF fake")
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="PDF loading requires PyMuPDF"):
            DocumentLoader().load_file(str(path))

    def test_load_directory(self, temp_dir):
        loader = DocumentLoader()
        docs = loader.load_directory(temp_dir)
        assert len(docs) >= 4  # 2 JSONL + 1 TXT + 1+ MD

    def test_load_directory_non_recursive_skips_nested_files(self, tmp_path):
        (tmp_path / "root.txt").write_text("root", encoding="utf-8")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "child.txt").write_text("child", encoding="utf-8")

        docs = DocumentLoader().load_directory(str(tmp_path), recursive=False)

        assert [doc.doc_id for doc in docs] == ["root"]

    def test_load_directory_continues_after_bad_supported_file(self, tmp_path, caplog):
        (tmp_path / "good.txt").write_text("good", encoding="utf-8")
        (tmp_path / "bad.json").write_text("[1]", encoding="utf-8")

        docs = DocumentLoader().load_directory(str(tmp_path))

        assert [doc.doc_id for doc in docs] == ["good"]
        assert "Failed to load" in caplog.text

    def test_load_directory_requires_directory(self, tmp_path):
        path = tmp_path / "file.txt"
        path.write_text("text", encoding="utf-8")

        with pytest.raises(NotADirectoryError):
            DocumentLoader().load_directory(str(path))

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
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"chunk_size": 0}, "chunk_size"),
            ({"chunk_size": 10, "chunk_overlap": 10}, "chunk_overlap"),
            ({"chunk_overlap": -1}, "chunk_overlap"),
            ({"min_chunk_size": -1}, "min_chunk_size"),
            ({"strategy": "unknown"}, "Unknown chunking strategy"),
        ],
    )
    def test_invalid_chunker_config_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            TextChunker(**kwargs)

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
        chunker = TextChunker(chunk_size=30, chunk_overlap=10, strategy="sentence", min_chunk_size=1)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert all(c.text for c in chunks)

    def test_sentence_chunk_splits_oversized_sentence(self):
        doc = RawDocument(doc_id="D1", title="T", content="没有句号的超长句子" * 20)
        chunker = TextChunker(chunk_size=60, chunk_overlap=10, strategy="sentence", min_chunk_size=1)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 2
        assert all(len(chunk.text) <= 60 for chunk in chunks)

    def test_sentence_chunk_keeps_overlap_sentences(self):
        text = "甲乙丙丁戊己。庚辛壬癸子丑。寅卯辰巳午未。申酉戌亥天地。"
        doc = RawDocument(doc_id="D1", title="T", content=text)
        chunker = TextChunker(chunk_size=22, chunk_overlap=9, strategy="sentence", min_chunk_size=1)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        assert chunks[0].text.split()[-1] == chunks[1].text.split()[0]

    def test_paragraph_chunk(self):
        text = "第一段内容。" * 10 + "\n\n" + "第二段内容。" * 10 + "\n\n" + "第三段内容。" * 10
        doc = RawDocument(doc_id="D1", title="T", content=text)
        chunker = TextChunker(chunk_size=200, strategy="paragraph")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_paragraph_chunk_flushes_before_oversized_paragraph(self):
        text = "短段落一。\n\n" + "无标点超长段落" * 12 + "\n\n短段落二。"
        doc = RawDocument(doc_id="D1", title="T", content=text)
        chunker = TextChunker(chunk_size=50, chunk_overlap=5, strategy="paragraph", min_chunk_size=1)

        chunks = chunker.chunk_document(doc)

        assert chunks[0].text == "短段落一。"
        assert chunks[-1].text == "短段落二。"
        assert len(chunks) > 3

    def test_paragraph_chunk_splits_oversized_paragraphs(self):
        text = "第一段没有句号但非常长" * 20 + "\n\n" + "第二段。"
        doc = RawDocument(doc_id="D1", title="T", content=text)
        chunker = TextChunker(chunk_size=80, chunk_overlap=10, strategy="paragraph", min_chunk_size=1)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 2
        assert all(len(chunk.text) <= 90 for chunk in chunks)
        assert [chunk.chunk_id for chunk in chunks] == [f"D1_c{i}" for i in range(len(chunks))]
        assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))

    def test_heading_chunk(self):
        doc = RawDocument(
            doc_id="D1",
            title="T",
            content="# 第一节\n" + "第一节内容。" * 10 + "\n## 第二节\n" + "第二节内容。" * 10,
        )
        chunker = TextChunker(chunk_size=80, strategy="heading", min_chunk_size=20)

        chunks = chunker.chunk_document(doc)

        assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
        assert chunks[0].text.startswith("# 第一节")
        assert chunks[1].text.startswith("## 第二节")

    def test_heading_chunk_splits_oversized_sections(self):
        doc = RawDocument(
            doc_id="D1",
            title="T",
            content="# 第一节\n" + "无标点超长章节内容" * 20 + "\n## 第二节\n短节。",
        )
        chunker = TextChunker(chunk_size=70, chunk_overlap=10, strategy="heading", min_chunk_size=1)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 3
        assert chunks[0].text.startswith("# 第一节")
        assert all(len(chunk.text) <= 80 for chunk in chunks)
        assert [chunk.chunk_id for chunk in chunks] == [f"D1_c{i}" for i in range(len(chunks))]

    def test_short_initial_section_merges_without_chunk_index_gap(self):
        doc = RawDocument(
            doc_id="D1",
            title="T",
            content="前言\n# 第一节\n" + "第一节内容。" * 8,
        )
        chunker = TextChunker(chunk_size=80, strategy="heading", min_chunk_size=20)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "D1_c0"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].text.startswith("前言\n# 第一节")

    def test_all_short_sections_are_preserved_as_one_chunk(self):
        doc = RawDocument(doc_id="D1", title="T", content="aa\n# bb\n## cc")
        chunker = TextChunker(chunk_size=80, strategy="heading", min_chunk_size=10)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].text == "aa\n# bb\n## cc"
        assert chunks[0].chunk_id == "D1_c0"

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
    class FakeRetriever:
        def __init__(self, config=None, model_name=None):
            self.config = config
            self.model_name = model_name
            self.documents = []

        def index_documents(self, documents):
            self.documents = documents

        def retrieve(self, query, top_k=10, **kwargs):
            return []

    def test_ingest_file(self):
        pipeline = IngestionPipeline(chunk_size=512)
        chunks = pipeline.ingest_file("data/verabench/corpus.jsonl")
        assert len(chunks) >= 33

    def test_ingest_documents_chunks_raw_documents(self):
        pipeline = IngestionPipeline(chunk_size=20, chunk_overlap=5, min_chunk_size=5)
        docs = [
            RawDocument(
                doc_id="D1",
                title="Title",
                content="alpha beta gamma delta epsilon",
                source="fixture",
                metadata={"date": "2026-06-18"},
            )
        ]

        chunks = pipeline.ingest_documents(docs)

        assert chunks
        assert chunks[0].doc_id == "D1"
        assert chunks[0].metadata["source"] == "fixture"
        assert chunks[0].metadata["date"] == "2026-06-18"

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

    def test_ingest_and_index_directory_flag(self, temp_dir):
        pipeline = IngestionPipeline(chunk_size=512)

        chunks, retriever = pipeline.ingest_and_index(
            temp_dir,
            retriever_type=" BM25 ",
            is_directory=True,
        )

        assert len(chunks) >= 2
        assert retriever.retrieve("测试文档", top_k=1)

    def test_build_retriever_index_rejects_empty_chunks(self):
        pipeline = IngestionPipeline()

        with pytest.raises(ValueError, match="zero chunks"):
            pipeline.build_retriever_index([])

    def test_build_retriever_index_rejects_unknown_type(self):
        pipeline = IngestionPipeline()
        chunk = Chunk(chunk_id="C1", doc_id="D1", text="text")

        with pytest.raises(ValueError, match="Unknown retriever type"):
            pipeline.build_retriever_index([chunk], retriever_type="unknown")

    @pytest.mark.parametrize(
        ("retriever_type", "module_name", "class_names"),
        [
            ("dense", "src.retriever.dense", ("DenseRetriever",)),
            ("faiss", "src.retriever.dense", ("FAISSRetriever",)),
            ("hybrid", "src.retriever.hybrid", ("HybridRetriever",)),
        ],
    )
    def test_build_retriever_index_non_bm25_branches_are_wired(
        self,
        monkeypatch,
        retriever_type,
        module_name,
        class_names,
    ):
        module = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, self.FakeRetriever)
        monkeypatch.setitem(sys.modules, module_name, module)

        pipeline = IngestionPipeline()
        chunk = Chunk(
            chunk_id="C1",
            doc_id="D1",
            text="retriever branch text",
            title="Branch",
            metadata={"source": "fixture"},
        )

        retriever = pipeline.build_retriever_index(
            [chunk],
            retriever_type=retriever_type,
            retriever_config={"model_name": "local-test-model"},
        )

        assert isinstance(retriever, self.FakeRetriever)
        assert retriever.documents == [chunk.to_index_doc()]
        if retriever_type in {"dense", "faiss"}:
            assert retriever.model_name == "local-test-model"

    def test_ingest_and_index_from_docs_preserves_metadata_and_extra_fields(self):
        pipeline = IngestionPipeline()

        chunks, retriever = pipeline.ingest_and_index_from_docs(
            [
                {
                    "doc_id": "D1",
                    "content": "alpha beta searchable",
                    "title": "Alpha",
                    "metadata": {"date": "2026-06-18"},
                    "source": "fixture",
                },
                {"id": "D2", "text": "metadata can be null", "metadata": None},
            ]
        )

        assert chunks[0].chunk_id == "D1"
        assert chunks[0].metadata == {"date": "2026-06-18", "source": "fixture"}
        assert chunks[1].metadata == {}
        assert retriever.doc_ids == ["D1", "D2"]
        assert retriever.doc_metadata[0]["date"] == "2026-06-18"

    @pytest.mark.parametrize(
        ("documents", "message"),
        [
            ([{"text": "missing id"}], "missing a non-empty id"),
            ([{"id": "D1"}], "missing non-empty text"),
            ([{"id": "D1", "text": "   "}], "missing non-empty text"),
            ([{"id": "D1", "text": "text", "metadata": "bad"}], "metadata"),
            ([{"id": "D1", "text": "one"}, {"id": "D1", "text": "two"}], "Duplicate"),
            (["bad"], "must be a mapping"),
        ],
    )
    def test_ingest_and_index_from_docs_rejects_bad_documents(self, documents, message):
        pipeline = IngestionPipeline()

        with pytest.raises(ValueError, match=message):
            pipeline.ingest_and_index_from_docs(documents)
