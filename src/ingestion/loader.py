"""Document loader for various file formats."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class RawDocument:
    """A raw document before chunking."""
    doc_id: str
    title: str
    content: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Load documents from various file formats."""

    def load_file(self, file_path: str) -> List[RawDocument]:
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

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[RawDocument]:
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        extensions = {".jsonl", ".json", ".txt", ".md", ".pdf"}
        docs = []

        pattern = "**/*" if recursive else "*"
        for fp in path.glob(pattern):
            if fp.is_file() and fp.suffix.lower() in extensions:
                try:
                    docs.extend(self.load_file(str(fp)))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", fp, e)

        return docs

    def _load_jsonl(self, path: Path) -> List[RawDocument]:
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                doc_id = data.get("doc_id", data.get("id", f"{path.stem}_{i}"))
                title = data.get("title", "")
                content = data.get("content", data.get("text", ""))
                source = data.get("source", "")
                metadata = {k: v for k, v in data.items()
                           if k not in ("doc_id", "id", "title", "content", "text", "source")}
                docs.append(RawDocument(
                    doc_id=doc_id, title=title, content=content,
                    source=source, metadata=metadata,
                ))
        return docs

    def _load_json(self, path: Path) -> List[RawDocument]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            docs = []
            for i, item in enumerate(data):
                doc_id = item.get("doc_id", item.get("id", f"{path.stem}_{i}"))
                title = item.get("title", "")
                content = item.get("content", item.get("text", ""))
                source = item.get("source", "")
                metadata = {k: v for k, v in item.items()
                           if k not in ("doc_id", "id", "title", "content", "text", "source")}
                docs.append(RawDocument(
                    doc_id=doc_id, title=title, content=content,
                    source=source, metadata=metadata,
                ))
            return docs

        # Single document
        doc_id = data.get("doc_id", data.get("id", path.stem))
        return [RawDocument(
            doc_id=doc_id,
            title=data.get("title", ""),
            content=data.get("content", data.get("text", "")),
            source=data.get("source", ""),
            metadata={k: v for k, v in data.items()
                     if k not in ("doc_id", "id", "title", "content", "text", "source")},
        )]

    def _load_text(self, path: Path) -> List[RawDocument]:
        content = path.read_text(encoding="utf-8")
        return [RawDocument(
            doc_id=path.stem,
            title=path.stem,
            content=content,
            source="file",
            metadata={"file_path": str(path), "format": "txt"},
        )]

    def _load_markdown(self, path: Path) -> List[Document if False else RawDocument]:
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

        docs = []
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

    def _load_pdf(self, path: Path) -> List[RawDocument]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PDF loading requires PyMuPDF. Install with: pip install pymupdf"
            )

        doc = fitz.open(str(path))
        pages = []
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
        doc.close()
        return pages if pages else [RawDocument(
            doc_id=path.stem, title=path.stem, content="",
            metadata={"file_path": str(path), "format": "pdf"},
        )]
