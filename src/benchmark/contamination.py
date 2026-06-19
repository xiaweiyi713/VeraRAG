"""Local contamination and overlap audit helpers for VeraBench."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import VeraBenchLoader

TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl"}


@dataclass(frozen=True)
class AuditText:
    item_id: str
    kind: str
    text: str


@dataclass(frozen=True)
class ReferenceText:
    source_id: str
    path: str
    text: str
    sha256: str


def audit_verabench_contamination(
    data_dir: str | Path | None = None,
    *,
    reference_paths: Iterable[str | Path] = (),
    near_threshold: float = 0.85,
    containment_threshold: float = 0.85,
    ngram_size: int = 13,
    min_exact_chars: int = 8,
    max_matches: int = 50,
) -> dict[str, Any]:
    """Audit benchmark text overlap against caller-supplied local references.

    The audit is intentionally local and evidence-producing: it can prove
    overlap with files the caller provides, but it does not claim absence of
    contamination in unknown model-training corpora.
    """
    if not 0.0 <= near_threshold <= 1.0:
        raise ValueError("near_threshold must be between 0 and 1")
    if not 0.0 <= containment_threshold <= 1.0:
        raise ValueError("containment_threshold must be between 0 and 1")
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least 2")
    if min_exact_chars < 1:
        raise ValueError("min_exact_chars must be positive")
    if max_matches < 0:
        raise ValueError("max_matches must be non-negative")

    loader = VeraBenchLoader(str(data_dir) if data_dir else None)
    benchmark = loader.load()
    audit_items = _benchmark_audit_texts(benchmark)
    references = _load_reference_texts(reference_paths)

    all_exact_matches = _exact_matches(
        audit_items,
        references,
        min_exact_chars=min_exact_chars,
    )
    all_near_matches = _near_matches(
        audit_items,
        references,
        near_threshold=near_threshold,
        containment_threshold=containment_threshold,
        ngram_size=ngram_size,
    )
    exact_matches = _limit_matches(all_exact_matches, max_matches)
    near_matches = _limit_matches(all_near_matches, max_matches)
    high_risk_kinds = {"question", "ground_truth_answer"}
    high_risk_exact = [
        match for match in all_exact_matches
        if match["kind"] in high_risk_kinds
    ]
    high_risk_near = [
        match for match in all_near_matches
        if match["kind"] in high_risk_kinds
    ]
    return {
        "schema_version": "verabench-contamination-audit-v1",
        "status": "complete" if references else "no_references",
        "data_dir": str(loader.data_dir),
        "method": {
            "exact": {
                "normalized_substring_min_chars": min_exact_chars,
            },
            "near_duplicate": {
                "character_ngram_jaccard_threshold": near_threshold,
                "character_ngram_item_containment_threshold": containment_threshold,
                "ngram_size": ngram_size,
            },
            "interpretation": (
                "This audit only checks the provided local reference files. "
                "No-overlap results are not proof of absence from model "
                "training corpora."
            ),
        },
        "coverage": {
            "benchmark_items": len(audit_items),
            "reference_texts": len(references),
            "reference_files": len({reference.path for reference in references}),
        },
        "reference_fingerprints": [
            {
                "path": reference.path,
                "source_id": reference.source_id,
                "sha256": reference.sha256,
                "characters": len(reference.text),
            }
            for reference in references
        ],
        "matches": {
            "exact": exact_matches,
            "near_duplicate": near_matches,
        },
        "summary": {
            "exact_matches": len(all_exact_matches),
            "near_duplicate_matches": len(all_near_matches),
            "returned_exact_matches": len(exact_matches),
            "returned_near_duplicate_matches": len(near_matches),
            "matches_truncated": (
                len(exact_matches) < len(all_exact_matches)
                or len(near_matches) < len(all_near_matches)
            ),
            "high_risk_exact_matches": len(high_risk_exact),
            "high_risk_near_duplicate_matches": len(high_risk_near),
            "matched_item_ids": sorted(
                {
                    str(match["item_id"])
                    for match in all_exact_matches + all_near_matches
                }
            ),
        },
    }


def _benchmark_audit_texts(benchmark: Any) -> list[AuditText]:
    rows: list[AuditText] = []
    for question in benchmark.questions:
        rows.append(AuditText(question.id, "question", question.question))
        rows.append(
            AuditText(
                question.id,
                "ground_truth_answer",
                question.ground_truth_answer,
            )
        )
        for evidence in question.evidence:
            rows.append(
                AuditText(
                    f"{question.id}:{evidence.evidence_id}",
                    "evidence_span",
                    evidence.text_span,
                )
            )
    for doc_id, document in sorted(benchmark.corpus.items()):
        rows.append(AuditText(doc_id, "document_title", document.title))
        rows.append(AuditText(doc_id, "document_content", document.content))
    return rows


def _load_reference_texts(paths: Iterable[str | Path]) -> list[ReferenceText]:
    references: list[ReferenceText] = []
    for path_like in paths:
        path = Path(path_like)
        files = _iter_reference_files(path)
        for file_path in files:
            references.extend(_read_reference_file(file_path))
    return references


def _iter_reference_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError(f"unsupported reference file type: {path}")
        return [path]
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in TEXT_SUFFIXES
    )


def _read_reference_file(path: Path) -> list[ReferenceText]:
    raw = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for index, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                rows.append((f"{path.name}:{index}", line))
                continue
            rows.extend(
                (f"{path.name}:{index}:{field}", value)
                for field, value in _string_leaves(payload)
            )
        return [
            ReferenceText(source_id, str(path), text, digest)
            for source_id, text in rows
            if text.strip()
        ]
    if suffix == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return [ReferenceText(path.name, str(path), raw, digest)]
        return [
            ReferenceText(f"{path.name}:{field}", str(path), text, digest)
            for field, text in _string_leaves(payload)
            if text.strip()
        ]
    return [ReferenceText(path.name, str(path), raw, digest)]


def _string_leaves(payload: Any, prefix: str = "$") -> list[tuple[str, str]]:
    if isinstance(payload, str):
        return [(prefix, payload)]
    if isinstance(payload, list):
        rows: list[tuple[str, str]] = []
        for index, item in enumerate(payload):
            rows.extend(_string_leaves(item, f"{prefix}[{index}]"))
        return rows
    if isinstance(payload, dict):
        rows = []
        for key, value in sorted(payload.items()):
            rows.extend(_string_leaves(value, f"{prefix}.{key}"))
        return rows
    return []


def _normalize(text: str) -> str:
    return _normalize_with_offsets(text)[0]


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    normalized_characters: list[str] = []
    original_offsets: list[int] = []
    for index, character in enumerate(text):
        if character.isalnum():
            normalized_characters.append(character.lower())
            original_offsets.append(index)
    return "".join(normalized_characters), original_offsets


def _ngrams(text: str, size: int) -> set[str]:
    normalized = _normalize(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {
        normalized[index:index + size]
        for index in range(len(normalized) - size + 1)
    }


def _exact_matches(
    audit_items: list[AuditText],
    references: list[ReferenceText],
    *,
    min_exact_chars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized_references = [
        (reference, *_normalize_with_offsets(reference.text))
        for reference in references
    ]
    for item in audit_items:
        normalized_item = _normalize(item.text)
        if len(normalized_item) < min_exact_chars:
            continue
        for reference, normalized_reference, reference_offsets in normalized_references:
            normalized_start = normalized_reference.find(normalized_item)
            if normalized_start >= 0:
                normalized_end = normalized_start + len(normalized_item)
                reference_start = reference_offsets[normalized_start]
                reference_end = reference_offsets[normalized_end - 1] + 1
                rows.append({
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "reference": reference.source_id,
                    "path": reference.path,
                    "normalized_chars": len(normalized_item),
                    "reference_start_char": reference_start,
                    "reference_end_char": reference_end,
                    "benchmark_excerpt": _excerpt(item.text),
                    "reference_excerpt": _excerpt_around(
                        reference.text,
                        reference_start,
                        reference_end,
                    ),
                })
    return rows


def _near_matches(
    audit_items: list[AuditText],
    references: list[ReferenceText],
    *,
    near_threshold: float,
    containment_threshold: float,
    ngram_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_grams = [
        (reference, _ngrams(reference.text, ngram_size))
        for reference in references
    ]
    for item in audit_items:
        normalized_item = _normalize(item.text)
        item_grams = _ngrams(item.text, ngram_size)
        if not item_grams:
            continue
        for reference, grams in reference_grams:
            intersection = len(item_grams & grams)
            union = item_grams | grams
            jaccard = intersection / len(union) if union else 1.0
            containment = intersection / len(item_grams)
            if jaccard >= near_threshold or containment >= containment_threshold:
                basis = _near_match_basis(
                    jaccard,
                    containment,
                    near_threshold=near_threshold,
                    containment_threshold=containment_threshold,
                )
                rows.append({
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "reference": reference.source_id,
                    "path": reference.path,
                    "jaccard": round(jaccard, 6),
                    "item_ngram_containment": round(containment, 6),
                    "match_basis": basis,
                    "benchmark_excerpt": _excerpt(item.text),
                    "reference_excerpt": _best_reference_ngram_excerpt(
                        item_grams,
                        reference.text,
                        ngram_size=ngram_size,
                        item_normalized_chars=len(normalized_item),
                    ),
                })
    return rows


def _near_match_basis(
    jaccard: float,
    containment: float,
    *,
    near_threshold: float,
    containment_threshold: float,
) -> str:
    passes_jaccard = jaccard >= near_threshold
    passes_containment = containment >= containment_threshold
    if passes_jaccard and passes_containment:
        return "jaccard_and_item_containment"
    if passes_containment:
        return "item_containment"
    return "jaccard"


def _limit_matches(rows: list[dict[str, Any]], max_matches: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -max(
                float(row.get("jaccard", 1.0)),
                float(row.get("item_ngram_containment", 0.0)),
            ),
            row["kind"],
            row["item_id"],
            row["path"],
            row["reference"],
        ),
    )[:max_matches]


def _excerpt(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _excerpt_around(
    text: str,
    start: int,
    end: int,
    *,
    limit: int = 160,
) -> str:
    compact_start = max(0, start - max(0, (limit - (end - start)) // 2))
    compact_end = min(len(text), compact_start + limit)
    compact_start = max(0, compact_end - limit)
    excerpt = text[compact_start:compact_end]
    prefix = "…" if compact_start > 0 else ""
    suffix = "…" if compact_end < len(text) else ""
    return prefix + " ".join(excerpt.split()) + suffix


def _best_reference_ngram_excerpt(
    item_grams: set[str],
    reference_text: str,
    *,
    ngram_size: int,
    item_normalized_chars: int,
) -> str:
    normalized_reference, reference_offsets = _normalize_with_offsets(reference_text)
    if not normalized_reference or not reference_offsets:
        return _excerpt(reference_text)
    if len(normalized_reference) < ngram_size:
        return _excerpt_around(
            reference_text,
            reference_offsets[0],
            reference_offsets[-1] + 1,
        )

    hit_positions = [
        index
        for index in range(len(normalized_reference) - ngram_size + 1)
        if normalized_reference[index:index + ngram_size] in item_grams
    ]
    if not hit_positions:
        return _excerpt(reference_text)

    window = max(item_normalized_chars, ngram_size)
    best_left = 0
    best_right = 0
    left = 0
    for right, position in enumerate(hit_positions):
        while position - hit_positions[left] > window:
            left += 1
        if right - left > best_right - best_left:
            best_left = left
            best_right = right

    normalized_start = hit_positions[best_left]
    normalized_end = min(
        len(reference_offsets),
        max(
            normalized_start + window,
            hit_positions[best_right] + ngram_size,
        ),
    )
    reference_start = reference_offsets[normalized_start]
    reference_end = reference_offsets[normalized_end - 1] + 1
    return _excerpt_around(reference_text, reference_start, reference_end)
