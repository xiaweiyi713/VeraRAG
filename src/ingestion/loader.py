"""Document loader for various file formats."""

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DOCUMENT_FIELDS = {"doc_id", "id", "title", "content", "text", "source"}


@dataclass
class RawDocument:
    """A raw document before chunking."""
    doc_id: str
    title: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Load documents from various file formats."""

    def load_file(self, file_path: str) -> list[RawDocument]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        loaders = {
            ".jsonl": self._load_jsonl,
            ".json": self._load_json,
            ".txt": self._load_text,
            ".md": self._load_markdown,
            ".pdf": self._load_pdf,
        }

        loader = loaders.get(suffix)
        if not loader:
            raise ValueError(f"Unsupported file format: {suffix}")

        return loader(path)

    def load_directory(self, dir_path: str, recursive: bool = True) -> list[RawDocument]:
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        extensions = {".jsonl", ".json", ".txt", ".md", ".pdf"}
        docs: list[RawDocument] = []

        pattern = "**/*" if recursive else "*"
        for fp in path.glob(pattern):
            if fp.is_file() and fp.suffix.lower() in extensions:
                try:
                    docs.extend(self.load_file(str(fp)))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", fp, e)

        return docs

    def _load_jsonl(self, path: Path) -> list[RawDocument]:
        docs: list[RawDocument] = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line_no = i + 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_no}: {exc.msg}"
                    ) from exc
                docs.append(self._document_from_mapping(data, path, f"{path.stem}_{i}", line_no))
        return docs

    def _load_json(self, path: Path) -> list[RawDocument]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [
                self._document_from_mapping(item, path, f"{path.stem}_{i}", i + 1)
                for i, item in enumerate(data)
            ]

        if isinstance(data, Mapping):
            return [self._document_from_mapping(data, path, path.stem)]

        raise ValueError(f"JSON document must be an object or array of objects: {path}")

    def _document_from_mapping(
        self,
        data: Any,
        path: Path,
        fallback_doc_id: str,
        item_no: int | None = None,
    ) -> RawDocument:
        if not isinstance(data, Mapping):
            location = f" item {item_no}" if item_no is not None else ""
            raise ValueError(f"Expected JSON object in {path}{location}")

        doc_id = data.get("doc_id", data.get("id", fallback_doc_id))
        title = data.get("title", "")
        content = data.get("content", data.get("text", ""))
        source = data.get("source", "")

        if not isinstance(content, str):
            location = f" item {item_no}" if item_no is not None else ""
            raise ValueError(f"Document content must be a string in {path}{location}")

        metadata = {k: v for k, v in data.items() if k not in _DOCUMENT_FIELDS}
        return RawDocument(
            doc_id=str(doc_id),
            title=str(title),
            content=content,
            source=str(source),
            metadata=metadata,
        )

    def _load_text(self, path: Path) -> list[RawDocument]:
        content = path.read_text(encoding="utf-8")
        return [RawDocument(
            doc_id=path.stem,
            title=path.stem,
            content=content,
            source="file",
            metadata={"file_path": str(path), "format": "txt"},
        )]

    def _load_markdown(self, path: Path) -> list[RawDocument]:
        content = path.read_text(encoding="utf-8")
        # Split by headings into sections
        sections = re.split(r'\n(?=#{1,3}\s)', content)

        if len(sections) <= 1:
            return [RawDocument(
                doc_id=path.stem,
                title=path.stem,
                content=content,
                source="file",
                metadata={"file_path": str(path), "format": "md"},
            )]

        docs: list[RawDocument] = []
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            # Extract heading as title
            heading_match = re.match(r'^#{1,3}\s+(.+)', section)
            section_title = heading_match.group(1).strip() if heading_match else f"Section {i+1}"
            docs.append(RawDocument(
                doc_id=f"{path.stem}_s{i}",
                title=section_title,
                content=section,
                source="file",
                metadata={"file_path": str(path), "format": "md", "section": i},
            ))
        return docs

    def _load_pdf(self, path: Path) -> list[RawDocument]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PDF loading requires PyMuPDF. Install with: pip install pymupdf"
            )

        pages: list[RawDocument] = []
        doc = fitz.open(str(path))
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    pages.append(RawDocument(
                        doc_id=f"{path.stem}_p{page_num}",
                        title=f"{path.stem} - Page {page_num + 1}",
                        content=text,
                        source="file",
                        metadata={
                            "file_path": str(path),
                            "format": "pdf",
                            "page": page_num + 1,
                        },
                    ))
        finally:
            doc.close()
        return pages if pages else [RawDocument(
            doc_id=path.stem, title=path.stem, content="", source="file",
            metadata={"file_path": str(path), "format": "pdf"},
        )]
