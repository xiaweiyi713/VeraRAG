"""Audit protocol and annotation packets for external conflict sets."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path, PureWindowsPath
from typing import Any

from .loader import CONFLICT_TYPES, VeraBenchLoader

EXTERNAL_CONFLICT_SCHEMA_VERSION = "external-conflict-annotation-v1"
EXTERNAL_CONFLICT_PACKET_VERSION = "external-conflict-annotation-packet-v1"
NO_CONFLICT_LABEL = "none"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _normalize_conflict_type(row: dict[str, Any]) -> str:
    if not bool(row.get("conflict_present")):
        return NO_CONFLICT_LABEL
    return str(row.get("conflict_type", ""))


def _validate_annotator_id(annotator_id: str) -> None:
    if not annotator_id or annotator_id != annotator_id.strip():
        raise ValueError("annotator ids must be non-empty without surrounding whitespace")
    if "/" in annotator_id or "\\" in annotator_id:
        raise ValueError("annotator ids must not contain path separators")
    if annotator_id in {".", ".."}:
        raise ValueError("annotator ids must not be relative path markers")


def _packet_member_path(packet_root: Path, filename: Any, context: str) -> Path:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"{context} must be a non-empty relative path")

    relative_path = Path(filename)
    windows_path = PureWindowsPath(filename)
    invalid_parts = {"", ".", ".."}
    if relative_path.is_absolute() or windows_path.is_absolute():
        raise ValueError(f"{context} must be relative to the packet directory")
    if any(part in invalid_parts for part in relative_path.parts) or any(
        part in invalid_parts for part in windows_path.parts
    ):
        raise ValueError(f"{context} must not contain relative path markers")

    packet_root_resolved = packet_root.resolve()
    candidate = (packet_root_resolved / relative_path).resolve()
    try:
        candidate.relative_to(packet_root_resolved)
    except ValueError as exc:
        raise ValueError(f"{context} resolves outside the packet directory") from exc
    return candidate


def build_external_annotation_packet(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    annotator_ids: list[str],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create blind annotation templates for a VeraBench-compatible dataset."""
    if not annotator_ids:
        raise ValueError("at least one annotator id is required")
    for annotator_id in annotator_ids:
        _validate_annotator_id(annotator_id)
    if len(annotator_ids) != len(set(annotator_ids)):
        raise ValueError("annotator ids must be unique")

    root = Path(data_dir).resolve()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    packet_path = out / "packet_manifest.json"
    if packet_path.exists() and not overwrite:
        raise FileExistsError(
            f"{packet_path} already exists; pass overwrite=True to replace it"
        )

    benchmark = VeraBenchLoader(str(root)).load()
    manifest_path = root / "manifest.json"
    source_manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    source_fingerprints = {
        "corpus_sha256": _sha256(root / "corpus.jsonl"),
        "questions_sha256": _sha256(root / "questions.jsonl"),
    }
    if manifest_path.exists():
        source_fingerprints["manifest_sha256"] = _sha256(manifest_path)

    template_files: dict[str, str] = {}
    for annotator_id in annotator_ids:
        rows = [
            _annotation_template_row(question, benchmark.corpus, annotator_id)
            for question in benchmark.questions
        ]
        filename = f"annotations_{annotator_id}.jsonl"
        _write_jsonl(out / filename, rows)
        template_files[annotator_id] = filename

    adjudication_rows = [
        _adjudication_template_row(question, benchmark.corpus)
        for question in benchmark.questions
    ]
    _write_jsonl(out / "adjudications_template.jsonl", adjudication_rows)
    readme = _annotation_packet_readme(source_manifest)
    (out / "README.md").write_text(readme, encoding="utf-8")

    packet = {
        "schema_version": EXTERNAL_CONFLICT_PACKET_VERSION,
        "source_data_dir": str(root),
        "dataset_id": source_manifest.get("dataset_id"),
        "source_fingerprints": source_fingerprints,
        "questions": len(benchmark.questions),
        "documents": len(benchmark.corpus),
        "annotator_ids": annotator_ids,
        "template_files": template_files,
        "adjudication_template_file": "adjudications_template.jsonl",
        "readme_file": "README.md",
        "blind_fields_omitted": [
            "ground_truth_answer",
            "ground_truth_claims",
            "expected_conflicts",
            "expected_behavior",
            "question_type",
            "evidence_category",
        ],
    }
    _write_json(packet_path, packet)
    return packet


def compile_external_annotation_packet(
    packet_dir: str | Path,
    output_dir: str | Path,
    *,
    adjudicator_id: str = "adjudicator",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compile completed annotation packet templates into audit-ready files."""
    packet_root = Path(packet_dir)
    out = Path(output_dir)
    manifest = _read_json(packet_root / "packet_manifest.json")
    if manifest.get("schema_version") != EXTERNAL_CONFLICT_PACKET_VERSION:
        raise ValueError(
            "packet_manifest.schema_version must be "
            f"{EXTERNAL_CONFLICT_PACKET_VERSION!r}"
        )

    out.mkdir(parents=True, exist_ok=True)
    output_files = [
        out / "corpus.jsonl",
        out / "questions.jsonl",
        out / "manifest.json",
        out / "annotations.jsonl",
        out / "adjudications.jsonl",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"compiled output files already exist: {joined}; "
            "pass overwrite=True to replace them"
        )

    source_root = _resolve_packet_source_root(packet_root, manifest)
    benchmark = VeraBenchLoader(str(source_root)).load()
    evidence_ids_by_question = {
        question.id: {evidence.evidence_id for evidence in question.evidence}
        for question in benchmark.questions
    }

    annotations = _compile_annotation_rows(
        packet_root,
        manifest,
        evidence_ids_by_question,
    )
    adjudications = _compile_adjudication_rows(
        packet_root,
        manifest,
        evidence_ids_by_question,
        adjudicator_id,
    )

    for filename in ("corpus.jsonl", "questions.jsonl", "manifest.json"):
        source_path = source_root / filename
        if not source_path.exists():
            raise FileNotFoundError(f"source dataset is missing {source_path}")
        shutil.copyfile(source_path, out / filename)
    _write_jsonl(out / "annotations.jsonl", annotations)
    _write_jsonl(out / "adjudications.jsonl", adjudications)

    compiled_manifest = {
        "schema_version": "external-conflict-compiled-annotations-v1",
        "packet_schema_version": manifest["schema_version"],
        "source_data_dir": str(source_root),
        "output_dir": str(out),
        "dataset_id": manifest.get("dataset_id"),
        "questions": len(benchmark.questions),
        "annotations": len(annotations),
        "adjudications": len(adjudications),
        "annotator_ids": manifest.get("annotator_ids", []),
        "adjudicator_id": adjudicator_id,
        "files": {
            "corpus": "corpus.jsonl",
            "questions": "questions.jsonl",
            "manifest": "manifest.json",
            "annotations": "annotations.jsonl",
            "adjudications": "adjudications.jsonl",
        },
    }
    _write_json(out / "compiled_manifest.json", compiled_manifest)
    return compiled_manifest


def _compile_annotation_rows(
    packet_root: Path,
    manifest: dict[str, Any],
    evidence_ids_by_question: dict[str, set[str]],
) -> list[dict[str, Any]]:
    template_files = manifest.get("template_files", {})
    if not isinstance(template_files, dict):
        raise ValueError("packet_manifest.template_files must be an object")
    rows: list[dict[str, Any]] = []
    for annotator_id, filename in sorted(
        template_files.items(), key=lambda item: str(item[0])
    ):
        annotations_path = _packet_member_path(
            packet_root,
            filename,
            f"packet_manifest.template_files[{annotator_id!r}]",
        )
        for index, row in enumerate(_read_jsonl(annotations_path), start=1):
            question_id = str(row.get("question_id", ""))
            label = row.get("label")
            if not isinstance(label, dict):
                raise ValueError(f"{filename}:{index}: label object is required")
            compiled = {
                "question_id": question_id,
                "annotator_id": str(row.get("annotator_id") or annotator_id),
                "conflict_present": label.get("conflict_present"),
                "conflict_type": label.get("conflict_type"),
                "evidence_pair": label.get("evidence_pair", []),
                "rationale": label.get("rationale", ""),
            }
            _validate_completed_label(
                compiled,
                evidence_ids_by_question,
                f"{filename}:{index}",
            )
            rows.append(compiled)
    return rows


def _resolve_packet_source_root(
    packet_root: Path,
    manifest: dict[str, Any],
) -> Path:
    source_root = Path(str(manifest["source_data_dir"]))
    candidates = [source_root]
    if not source_root.is_absolute():
        candidates.append(packet_root / source_root)
    for candidate in candidates:
        if (candidate / "corpus.jsonl").exists() and (
            candidate / "questions.jsonl"
        ).exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "packet source_data_dir does not contain corpus.jsonl and questions.jsonl: "
        f"{source_root}"
    )


def _compile_adjudication_rows(
    packet_root: Path,
    manifest: dict[str, Any],
    evidence_ids_by_question: dict[str, set[str]],
    adjudicator_id: str,
) -> list[dict[str, Any]]:
    filename = manifest.get("adjudication_template_file", "")
    if not filename:
        raise ValueError("packet_manifest.adjudication_template_file is required")
    rows: list[dict[str, Any]] = []
    adjudication_path = _packet_member_path(
        packet_root,
        filename,
        "packet_manifest.adjudication_template_file",
    )
    for index, row in enumerate(_read_jsonl(adjudication_path), start=1):
        question_id = str(row.get("question_id", ""))
        label = row.get("gold_label")
        if not isinstance(label, dict):
            raise ValueError(f"{filename}:{index}: gold_label object is required")
        compiled = {
            "question_id": question_id,
            "adjudicator_id": str(row.get("adjudicator_id") or adjudicator_id),
            "gold_conflict_present": label.get("gold_conflict_present"),
            "gold_conflict_type": label.get("gold_conflict_type"),
            "evidence_pair": label.get("evidence_pair", []),
            "rationale": label.get("rationale", ""),
        }
        _validate_completed_label(
            {
                "question_id": compiled["question_id"],
                "conflict_present": compiled["gold_conflict_present"],
                "conflict_type": compiled["gold_conflict_type"],
                "evidence_pair": compiled["evidence_pair"],
            },
            evidence_ids_by_question,
            f"{filename}:{index}",
        )
        rows.append(compiled)
    return rows


def _validate_completed_label(
    row: dict[str, Any],
    evidence_ids_by_question: dict[str, set[str]],
    context: str,
) -> None:
    question_id = str(row.get("question_id", ""))
    if question_id not in evidence_ids_by_question:
        raise ValueError(f"{context}: unknown question_id {question_id!r}")
    errors: list[str] = []
    _validate_label_row(
        row,
        evidence_ids_by_question[question_id],
        context,
        errors,
    )
    if errors:
        raise ValueError("; ".join(errors))


def _annotation_template_row(
    question: Any,
    corpus: dict[str, Any],
    annotator_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_CONFLICT_PACKET_VERSION,
        "question_id": question.id,
        "annotator_id": annotator_id,
        "question": question.question,
        "evidence": [
            {
                "evidence_id": evidence.evidence_id,
                "doc_id": evidence.doc_id,
                "title": corpus[evidence.doc_id].title,
                "source": corpus[evidence.doc_id].source,
                "date": corpus[evidence.doc_id].date,
                "text_span": evidence.text_span,
            }
            for evidence in question.evidence
        ],
        "label": {
            "conflict_present": None,
            "conflict_type": "",
            "evidence_pair": [],
            "rationale": "",
        },
    }


def _adjudication_template_row(question: Any, corpus: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_CONFLICT_PACKET_VERSION,
        "question_id": question.id,
        "adjudicator_id": "",
        "question": question.question,
        "evidence": [
            {
                "evidence_id": evidence.evidence_id,
                "doc_id": evidence.doc_id,
                "title": corpus[evidence.doc_id].title,
                "source": corpus[evidence.doc_id].source,
                "date": corpus[evidence.doc_id].date,
                "text_span": evidence.text_span,
            }
            for evidence in question.evidence
        ],
        "gold_label": {
            "gold_conflict_present": None,
            "gold_conflict_type": "",
            "evidence_pair": [],
            "rationale": "",
        },
    }


def _annotation_packet_readme(source_manifest: dict[str, Any]) -> str:
    dataset_id = source_manifest.get("dataset_id", "external-conflict-set")
    allowed_types = ", ".join(CONFLICT_TYPES)
    return (
        f"# External Conflict Annotation Packet: {dataset_id}\n\n"
        "Annotate each question independently. Use only the visible question and "
        "evidence snippets in the JSONL row.\n\n"
        "## Label Rules\n\n"
        "- Set `conflict_present` to `true` only when two provided evidence "
        "snippets make incompatible claims about the same question-relevant "
        "fact.\n"
        "- Set `conflict_present` to `false` when the snippets are compatible, "
        "one snippet is merely incomplete, or the disagreement is irrelevant "
        "to the question.\n"
        f"- Allowed conflict types: `{allowed_types}`.\n"
        "- Use `evidence_pair` for the smallest pair of evidence ids that "
        "supports the label.\n"
        "- Write a short rationale grounded in the provided snippets.\n\n"
        "## Finalization\n\n"
        "Convert each template row into the audit schema before validation: "
        "`question_id`, `annotator_id`, `conflict_present`, `conflict_type`, "
        "`evidence_pair`, and `rationale`. No-conflict rows must use "
        "`conflict_type: \"none\"`.\n"
    )


def _cohen_kappa(left: list[str], right: list[str]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("kappa label lists must have equal length")
    if not left:
        return {
            "cohen_kappa": None,
            "observed_agreement": None,
            "comparisons": 0,
        }

    total = len(left)
    observed = sum(
        first == second
        for first, second in zip(left, right, strict=True)
    ) / total
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in set(left_counts) | set(right_counts)
    )
    if expected == 1.0:
        kappa = 1.0 if observed == 1.0 else 0.0
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {
        "cohen_kappa": round(kappa, 6),
        "observed_agreement": round(observed, 6),
        "comparisons": total,
    }


def _agreement_report(
    annotations_by_question: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    conflict_left: list[str] = []
    conflict_right: list[str] = []
    type_left: list[str] = []
    type_right: list[str] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    disagreement_questions: set[str] = set()

    for question_id, labels_by_annotator in sorted(annotations_by_question.items()):
        for left_id, right_id in combinations(sorted(labels_by_annotator), 2):
            left_row = labels_by_annotator[left_id]
            right_row = labels_by_annotator[right_id]
            left_conflict = "conflict" if bool(left_row.get("conflict_present")) else NO_CONFLICT_LABEL
            right_conflict = "conflict" if bool(right_row.get("conflict_present")) else NO_CONFLICT_LABEL
            left_type = _normalize_conflict_type(left_row)
            right_type = _normalize_conflict_type(right_row)

            conflict_left.append(left_conflict)
            conflict_right.append(right_conflict)
            type_left.append(left_type)
            type_right.append(right_type)
            pair_counts[(left_id, right_id)] += 1
            if left_conflict != right_conflict or left_type != right_type:
                disagreement_questions.add(question_id)

    return {
        "conflict_present": _cohen_kappa(conflict_left, conflict_right),
        "conflict_type": _cohen_kappa(type_left, type_right),
        "annotator_pairs": [
            {
                "annotators": list(pair),
                "shared_questions": count,
            }
            for pair, count in sorted(pair_counts.items())
        ],
        "disagreement_questions": sorted(disagreement_questions),
    }


def audit_external_conflict_set(
    data_dir: str | Path,
    *,
    internal_data_dir: str | Path | None = None,
    min_questions: int = 1,
    min_annotators_per_question: int = 2,
    min_conflict_kappa: float = 0.6,
) -> dict[str, Any]:
    """Validate an external conflict set and compute annotation agreement."""
    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    annotations_path = root / "annotations.jsonl"
    adjudications_path = root / "adjudications.jsonl"
    required_files = [
        root / "corpus.jsonl",
        root / "questions.jsonl",
        manifest_path,
        annotations_path,
        adjudications_path,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        return {
            "valid": False,
            "schema_version": EXTERNAL_CONFLICT_SCHEMA_VERSION,
            "data_dir": str(root),
            "errors": [f"missing required file: {path}" for path in missing],
        }

    manifest = _read_json(manifest_path)
    annotations = _read_jsonl(annotations_path)
    adjudications = _read_jsonl(adjudications_path)
    benchmark = VeraBenchLoader(str(root)).load()
    question_ids = {question.id for question in benchmark.questions}
    evidence_ids_by_question = {
        question.id: {evidence.evidence_id for evidence in question.evidence}
        for question in benchmark.questions
    }
    expected_conflict_types_by_question = {
        question.id: {
            conflict.conflict_type
            for conflict in question.expected_conflicts
        }
        for question in benchmark.questions
    }
    errors: list[str] = []

    if manifest.get("schema_version") != EXTERNAL_CONFLICT_SCHEMA_VERSION:
        errors.append(
            "manifest.schema_version must be "
            f"{EXTERNAL_CONFLICT_SCHEMA_VERSION!r}"
        )
    if not manifest.get("dataset_id"):
        errors.append("manifest.dataset_id is required")
    if not manifest.get("annotation_protocol"):
        errors.append("manifest.annotation_protocol is required")

    annotations_by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    annotator_ids: set[str] = set()
    for index, row in enumerate(annotations, start=1):
        question_id = str(row.get("question_id", ""))
        annotator_id = str(row.get("annotator_id", ""))
        if question_id not in question_ids:
            errors.append(f"annotations row {index}: unknown question_id {question_id!r}")
            continue
        if not annotator_id:
            errors.append(f"annotations row {index}: annotator_id is required")
            continue
        if annotator_id in annotations_by_question[question_id]:
            errors.append(
                f"annotations row {index}: duplicate label for "
                f"{question_id}/{annotator_id}"
            )
        annotator_ids.add(annotator_id)
        annotations_by_question[question_id][annotator_id] = row
        _validate_label_row(
            row,
            evidence_ids_by_question[question_id],
            f"annotations row {index}",
            errors,
        )

    adjudication_by_question: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(adjudications, start=1):
        question_id = str(row.get("question_id", ""))
        if question_id not in question_ids:
            errors.append(f"adjudications row {index}: unknown question_id {question_id!r}")
            continue
        if question_id in adjudication_by_question:
            errors.append(f"adjudications row {index}: duplicate adjudication for {question_id}")
        adjudication_by_question[question_id] = row
        gold_row = {
            "conflict_present": row.get("gold_conflict_present"),
            "conflict_type": row.get("gold_conflict_type"),
            "evidence_pair": row.get("evidence_pair"),
        }
        _validate_label_row(
            gold_row,
            evidence_ids_by_question[question_id],
            f"adjudications row {index}",
            errors,
        )
        expected_types = expected_conflict_types_by_question[question_id]
        gold_present = bool(row.get("gold_conflict_present"))
        gold_type = _normalize_conflict_type(gold_row)
        if gold_present and gold_type not in expected_types:
            errors.append(
                f"adjudications row {index}: gold conflict type {gold_type!r} "
                f"is absent from question.expected_conflicts"
            )
        if not gold_present and expected_types:
            errors.append(
                f"adjudications row {index}: no-conflict gold conflicts with "
                "question.expected_conflicts"
            )

    questions_with_required_annotators = sum(
        len(labels) >= min_annotators_per_question
        for labels in annotations_by_question.values()
    )
    if len(benchmark.questions) < min_questions:
        errors.append(
            f"dataset has {len(benchmark.questions)} questions; "
            f"requires at least {min_questions}"
        )
    missing_annotation_questions = sorted(question_ids - set(annotations_by_question))
    if missing_annotation_questions:
        errors.append(
            "questions missing annotations: "
            + ", ".join(missing_annotation_questions)
        )
    under_annotated = sorted(
        question_id
        for question_id in question_ids
        if len(annotations_by_question.get(question_id, {})) < min_annotators_per_question
    )
    if under_annotated:
        errors.append(
            "questions below required annotator count: "
            + ", ".join(under_annotated)
        )
    missing_adjudications = sorted(question_ids - set(adjudication_by_question))
    if missing_adjudications:
        errors.append(
            "questions missing adjudications: "
            + ", ".join(missing_adjudications)
        )

    agreement = _agreement_report(annotations_by_question)
    conflict_kappa = agreement["conflict_present"]["cohen_kappa"]
    if conflict_kappa is None:
        errors.append("not enough paired annotations to compute conflict kappa")
    elif conflict_kappa < min_conflict_kappa:
        errors.append(
            f"conflict_present kappa {conflict_kappa} is below {min_conflict_kappa}"
        )

    adjudication_overrides = 0
    for question_id, gold in adjudication_by_question.items():
        gold_type = _normalize_conflict_type({
            "conflict_present": gold.get("gold_conflict_present"),
            "conflict_type": gold.get("gold_conflict_type"),
        })
        if any(
            _normalize_conflict_type(label) != gold_type
            for label in annotations_by_question.get(question_id, {}).values()
        ):
            adjudication_overrides += 1

    fingerprints = {
        name: _sha256(root / name)
        for name in (
            "corpus.jsonl",
            "questions.jsonl",
            "manifest.json",
            "annotations.jsonl",
            "adjudications.jsonl",
        )
    }
    internal_fingerprints = None
    if internal_data_dir is not None:
        internal_root = Path(internal_data_dir)
        internal_fingerprints = {
            "corpus_sha256": _sha256(internal_root / "corpus.jsonl"),
            "questions_sha256": _sha256(internal_root / "questions.jsonl"),
        }
        if fingerprints["questions.jsonl"] == internal_fingerprints["questions_sha256"]:
            errors.append("external questions fingerprint matches the internal benchmark")

    return {
        "valid": not errors,
        "schema_version": EXTERNAL_CONFLICT_SCHEMA_VERSION,
        "data_dir": str(root),
        "dataset": {
            "dataset_id": manifest.get("dataset_id"),
            "source": manifest.get("source"),
            "license": manifest.get("license"),
            "annotation_protocol": manifest.get("annotation_protocol"),
            "fingerprints": fingerprints,
            "internal_fingerprints": internal_fingerprints,
        },
        "minimums": {
            "questions": min_questions,
            "annotators_per_question": min_annotators_per_question,
            "conflict_present_kappa": min_conflict_kappa,
        },
        "coverage": {
            "questions": len(benchmark.questions),
            "documents": len(benchmark.corpus),
            "annotations": len(annotations),
            "adjudications": len(adjudications),
            "annotators": sorted(annotator_ids),
            "questions_with_required_annotators": questions_with_required_annotators,
            "questions_with_adjudication": len(adjudication_by_question),
            "gold_conflict_questions": sum(
                bool(row.get("gold_conflict_present"))
                for row in adjudication_by_question.values()
            ),
        },
        "agreement": agreement,
        "adjudication": {
            "overrides": adjudication_overrides,
        },
        "errors": errors,
    }


def _validate_label_row(
    row: dict[str, Any],
    evidence_ids: set[str],
    context: str,
    errors: list[str],
) -> None:
    if not isinstance(row.get("conflict_present"), bool):
        errors.append(f"{context}: conflict_present must be boolean")
        return
    conflict_type = str(row.get("conflict_type", ""))
    if bool(row["conflict_present"]):
        if conflict_type not in CONFLICT_TYPES:
            errors.append(f"{context}: unknown conflict_type {conflict_type!r}")
        pair = row.get("evidence_pair")
        if not isinstance(pair, list) or len(pair) != 2:
            errors.append(f"{context}: conflict labels require a 2-item evidence_pair")
        elif any(evidence_id not in evidence_ids for evidence_id in pair):
            errors.append(f"{context}: evidence_pair references unknown evidence id")
    elif conflict_type != NO_CONFLICT_LABEL:
        errors.append(f"{context}: no-conflict labels must use conflict_type 'none'")
