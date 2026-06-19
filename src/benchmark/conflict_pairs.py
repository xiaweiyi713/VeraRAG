"""Build pairwise conflict training examples from VeraBench annotations."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from .loader import (
    BenchmarkQuestion,
    VeraBench,
    evidence_dependency_groups,
    load_verabench,
)

SPLIT_STRATEGY = "shared-evidence-component-stratified-v1"
_SPLITS = ("train", "val", "test")
_ALLOCATION_ORDER = ("val", "test", "train")


def _validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Expected 0 < train_ratio and 0 <= val_ratio with "
            "train_ratio + val_ratio < 1"
        )


@dataclass(frozen=True)
class ConflictPairExample:
    """One pairwise training/evaluation example for conflict detection."""

    id: str
    question_id: str
    paired_question_id: str
    question_type: str
    dependency_group: str
    paired_dependency_group: str
    text_a: str
    text_b: str
    label: int
    conflict_type: str
    evidence_id_a: str
    evidence_id_b: str
    doc_id_a: str
    doc_id_b: str
    category_a: str
    category_b: str
    split: str
    sample_source: str = "gold_or_in_question"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_quotas(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in _SPLITS}
    quotas = {split: int(raw[split]) for split in _SPLITS}
    remaining = total - sum(quotas.values())
    priority = {"val": 0, "test": 1, "train": 2}
    for split in sorted(
        _SPLITS,
        key=lambda item: (-(raw[item] - quotas[item]), priority[item]),
    )[:remaining]:
        quotas[split] += 1

    active = [split for split in _SPLITS if ratios[split] > 0]
    if total >= len(active):
        for split in active:
            if quotas[split] > 0:
                continue
            donor = max(
                (
                    candidate
                    for candidate in active
                    if quotas[candidate] > 1
                ),
                key=lambda candidate: (
                    quotas[candidate] - raw[candidate],
                    quotas[candidate],
                ),
            )
            quotas[donor] -= 1
            quotas[split] += 1
    return quotas


def _assign_group_class(
    group_ids: list[str],
    ratios: dict[str, float],
) -> dict[str, str]:
    quotas = _split_quotas(len(group_ids), ratios)
    ordered = sorted(
        group_ids,
        key=lambda group_id: hashlib.sha1(group_id.encode("utf-8")).hexdigest(),
    )
    assigned: dict[str, str] = {}
    offset = 0
    for split in _ALLOCATION_ORDER:
        end = offset + quotas[split]
        for group_id in ordered[offset:end]:
            assigned[group_id] = split
        offset = end
    return assigned


def _dependency_group_split_maps(
    questions: list[BenchmarkQuestion],
    train_ratio: float,
    val_ratio: float,
) -> tuple[dict[str, str], dict[str, str]]:
    _validate_split_ratios(train_ratio, val_ratio)
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    dependency_group_by_question = evidence_dependency_groups(questions)
    questions_by_group: dict[str, list[BenchmarkQuestion]] = {}
    for question in questions:
        group_id = dependency_group_by_question[question.id]
        questions_by_group.setdefault(group_id, []).append(question)

    positive_groups = [
        group_id
        for group_id, grouped_questions in questions_by_group.items()
        if any(question.expected_conflicts for question in grouped_questions)
    ]
    negative_groups = [
        group_id
        for group_id in questions_by_group
        if group_id not in positive_groups
    ]
    split_by_group = {
        **_assign_group_class(positive_groups, ratios),
        **_assign_group_class(negative_groups, ratios),
    }
    split_by_question = {
        question.id: split_by_group[dependency_group_by_question[question.id]]
        for question in questions
    }
    return split_by_question, split_by_group


def _conflict_lookup(question: BenchmarkQuestion) -> dict[frozenset[str], str]:
    lookup: dict[frozenset[str], str] = {}
    for conflict in question.expected_conflicts:
        if len(conflict.pair) != 2:
            continue
        lookup[frozenset(conflict.pair)] = conflict.conflict_type
    return lookup


def _question_tags(question: BenchmarkQuestion) -> set[str]:
    """Return topical tags, excluding task-shape labels."""
    excluded = {"冲突", "误导", "事实核查", "数值", "实体", "学术"}
    return {tag for tag in question.tags if tag not in excluded}


def _overlap_score(tags_a: set[str], tags_b: set[str]) -> float:
    if not tags_a or not tags_b:
        return 0.0
    return len(tags_a & tags_b) / len(tags_a | tags_b)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:%|％|°C|MW|GW|nm|万|亿|年|个|辆|吨|公里)?", text))


def _years(text: str) -> set[str]:
    return set(re.findall(r"(?:18|19|20)\d{2}年?", text))


def _has_mismatched_values(text_a: str, text_b: str) -> bool:
    years_a = _years(text_a)
    years_b = _years(text_b)
    if years_a and years_b and years_a != years_b:
        return True
    numbers_a = _numbers(text_a)
    numbers_b = _numbers(text_b)
    return bool(numbers_a and numbers_b and numbers_a != numbers_b)


def _infer_weak_conflict_type(text_a: str, text_b: str) -> str:
    years_a = _years(text_a)
    years_b = _years(text_b)
    if years_a and years_b and years_a != years_b:
        return "temporal_conflict"
    if _has_mismatched_values(text_a, text_b):
        return "numeric_conflict"
    scope_markers = ("所有", "全部", "仅", "只有", "不再", "不同", "并不", "不是")
    if any(marker in text_a or marker in text_b for marker in scope_markers):
        return "scope_conflict"
    return "definitional_conflict"


def _split_self_conflict_text(text: str) -> tuple[str, str]:
    """Split a single annotated evidence span into two training sides."""
    clauses = [part.strip() for part in re.split(r"[。；;.!?！？]+", text) if part.strip()]
    if len(clauses) >= 2:
        best_pair = (clauses[0], clauses[1])
        best_score = -1
        for left, right in combinations(clauses, 2):
            score = 0
            score += int(_has_mismatched_values(left, right)) * 3
            score += int(any(marker in left + right for marker in ("但", "然而", "不过", "并不", "不是", "质疑"))) * 2
            score += int(bool(_text_tokens(left) & _text_tokens(right)))
            if score > best_score:
                best_pair = (left, right)
                best_score = score
        return best_pair

    midpoint = max(1, len(text) // 2)
    return text[:midpoint].strip(), text[midpoint:].strip() or text.strip()


def _text_tokens(text: str) -> set[str]:
    """Tiny topical-token helper used only to rank self-conflict clause splits."""
    return set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text))


def _make_example(
    *,
    question: BenchmarkQuestion,
    paired_question_id: str,
    dependency_group: str,
    paired_dependency_group: str,
    text_a: str,
    text_b: str,
    label: int,
    conflict_type: str,
    evidence_id_a: str,
    evidence_id_b: str,
    doc_id_a: str,
    doc_id_b: str,
    category_a: str,
    category_b: str,
    split: str,
    sample_source: str,
    rationale: str,
    id_suffix: str | None = None,
) -> ConflictPairExample:
    suffix = id_suffix or f"{evidence_id_a}:{evidence_id_b}"
    return ConflictPairExample(
        id=f"{question.id}:{suffix}",
        question_id=question.id,
        paired_question_id=paired_question_id,
        question_type=question.type,
        dependency_group=dependency_group,
        paired_dependency_group=paired_dependency_group,
        text_a=text_a,
        text_b=text_b,
        label=label,
        conflict_type=conflict_type,
        evidence_id_a=evidence_id_a,
        evidence_id_b=evidence_id_b,
        doc_id_a=doc_id_a,
        doc_id_b=doc_id_b,
        category_a=category_a,
        category_b=category_b,
        split=split,
        sample_source=sample_source,
        rationale=rationale,
    )


def build_conflict_pair_examples(
    benchmark: VeraBench | None = None,
    *,
    data_dir: str | None = None,
    max_negative_per_question: int | None = 4,
    max_hard_negative_per_question: int = 2,
    max_weak_positive_per_question: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> list[ConflictPairExample]:
    """Create deterministic pairwise examples from VeraBench evidence refs.

    Positive examples are annotated ``expected_conflicts`` plus conservative
    weak positives when a supporting/conflicting pair has mismatched dates or
    numbers. Negative examples are same-question non-conflicts plus cross-question
    hard negatives with topical tag overlap.
    """
    bench = benchmark or load_verabench(data_dir)
    split_by_question, _ = _dependency_group_split_maps(
        bench.questions,
        train_ratio,
        val_ratio,
    )
    dependency_group_by_question = evidence_dependency_groups(bench.questions)
    examples: list[ConflictPairExample] = []
    seen_ids: set[str] = set()
    evidence_index = [
        (question, evidence, _question_tags(question))
        for question in bench.questions
        for evidence in question.evidence
        if evidence.text_span
    ]

    for question in bench.questions:
        if len(question.evidence) < 2:
            split = split_by_question[question.id]
            dependency_group = dependency_group_by_question[question.id]
            conflict_lookup = _conflict_lookup(question)
            for pair_key, gold_conflict_type in conflict_lookup.items():
                if len(pair_key) != 1:
                    continue
                evidence_id = next(iter(pair_key))
                evidence = next((item for item in question.evidence if item.evidence_id == evidence_id), None)
                if evidence is None:
                    continue
                text_a, text_b = _split_self_conflict_text(evidence.text_span)
                example = _make_example(
                    question=question,
                    paired_question_id=question.id,
                    dependency_group=dependency_group,
                    paired_dependency_group=dependency_group,
                    text_a=text_a,
                    text_b=text_b,
                    label=1,
                    conflict_type=gold_conflict_type,
                    evidence_id_a=evidence.evidence_id,
                    evidence_id_b=evidence.evidence_id,
                    doc_id_a=evidence.doc_id,
                    doc_id_b=evidence.doc_id,
                    category_a=evidence.category,
                    category_b=evidence.category,
                    split=split,
                    sample_source="gold_self_conflict",
                    rationale="Gold expected_conflicts references the same evidence span; text was split into two clauses.",
                    id_suffix=f"{evidence.evidence_id}:{evidence.evidence_id}:self",
                )
                if example.id not in seen_ids:
                    examples.append(example)
                    seen_ids.add(example.id)
            continue

        conflict_lookup = _conflict_lookup(question)
        emitted_gold_self: set[str] = set()
        negatives_added = 0
        weak_positives_added = 0
        split = split_by_question[question.id]
        dependency_group = dependency_group_by_question[question.id]

        for pair_key, gold_conflict_type in conflict_lookup.items():
            if len(pair_key) != 1:
                continue
            evidence_id = next(iter(pair_key))
            evidence = next((item for item in question.evidence if item.evidence_id == evidence_id), None)
            if evidence is None:
                continue
            text_a, text_b = _split_self_conflict_text(evidence.text_span)
            example = _make_example(
                question=question,
                paired_question_id=question.id,
                dependency_group=dependency_group,
                paired_dependency_group=dependency_group,
                text_a=text_a,
                text_b=text_b,
                label=1,
                conflict_type=gold_conflict_type,
                evidence_id_a=evidence.evidence_id,
                evidence_id_b=evidence.evidence_id,
                doc_id_a=evidence.doc_id,
                doc_id_b=evidence.doc_id,
                category_a=evidence.category,
                category_b=evidence.category,
                split=split,
                sample_source="gold_self_conflict",
                rationale="Gold expected_conflicts references the same evidence span; text was split into two clauses.",
                id_suffix=f"{evidence.evidence_id}:{evidence.evidence_id}:self",
            )
            if example.id not in seen_ids:
                examples.append(example)
                seen_ids.add(example.id)
                emitted_gold_self.add(evidence_id)

        for left, right in combinations(question.evidence, 2):
            pair_key = frozenset({left.evidence_id, right.evidence_id})
            pair_conflict_type = conflict_lookup.get(pair_key)
            is_positive = pair_conflict_type is not None
            sample_source = "gold_conflict" if is_positive else "in_question_negative"
            rationale = "Gold expected_conflicts pair." if is_positive else "Same-question evidence pair not annotated as a conflict."
            if not is_positive and max_weak_positive_per_question > weak_positives_added:
                has_conflict_category = {left.category, right.category} & {"conflicting", "outdated"}
                has_support_category = {left.category, right.category} & {"supporting", "partial"}
                if has_conflict_category and has_support_category and _has_mismatched_values(left.text_span, right.text_span):
                    pair_conflict_type = _infer_weak_conflict_type(left.text_span, right.text_span)
                    is_positive = True
                    sample_source = "weak_conflict"
                    rationale = "Heuristic weak label: conflicting/outdated evidence paired with supporting evidence and mismatched values."
                    weak_positives_added += 1
            if not is_positive:
                if max_negative_per_question is not None and negatives_added >= max_negative_per_question:
                    continue
                negatives_added += 1

            example = _make_example(
                question=question,
                paired_question_id=question.id,
                dependency_group=dependency_group,
                paired_dependency_group=dependency_group,
                text_a=left.text_span,
                text_b=right.text_span,
                label=1 if is_positive else 0,
                conflict_type=pair_conflict_type or "none",
                evidence_id_a=left.evidence_id,
                evidence_id_b=right.evidence_id,
                doc_id_a=left.doc_id,
                doc_id_b=right.doc_id,
                category_a=left.category,
                category_b=right.category,
                split=split,
                sample_source=sample_source,
                rationale=rationale,
            )
            if example.id not in seen_ids:
                examples.append(example)
                seen_ids.add(example.id)

        if max_hard_negative_per_question <= 0:
            continue
        added_hard = 0
        tags = _question_tags(question)
        if not tags:
            continue
        candidates = []
        for left in question.evidence:
            if left.evidence_id in emitted_gold_self:
                continue
            for other_question, other_evidence, other_tags in evidence_index:
                if other_question.id == question.id:
                    continue
                if split_by_question[other_question.id] != split:
                    continue
                if left.doc_id == other_evidence.doc_id:
                    continue
                score = _overlap_score(tags, other_tags)
                if score <= 0:
                    continue
                if left.category not in {"supporting", "partial"} or other_evidence.category not in {"supporting", "partial"}:
                    continue
                candidates.append((score, left, other_question, other_evidence))
        candidates.sort(key=lambda item: (-item[0], item[1].evidence_id, item[2].id, item[3].evidence_id))
        for score, left, other_question, other_evidence in candidates:
            if added_hard >= max_hard_negative_per_question:
                break
            example = _make_example(
                question=question,
                paired_question_id=other_question.id,
                dependency_group=dependency_group,
                paired_dependency_group=dependency_group_by_question[
                    other_question.id
                ],
                text_a=left.text_span,
                text_b=other_evidence.text_span,
                label=0,
                conflict_type="none",
                evidence_id_a=left.evidence_id,
                evidence_id_b=other_evidence.evidence_id,
                doc_id_a=left.doc_id,
                doc_id_b=other_evidence.doc_id,
                category_a=left.category,
                category_b=other_evidence.category,
                split=split,
                sample_source="hard_negative",
                rationale=f"Cross-question hard negative with topical tag overlap {score:.2f} against {other_question.id}.",
                id_suffix=f"{left.evidence_id}:hard:{other_question.id}:{other_evidence.evidence_id}",
            )
            if example.id in seen_ids:
                continue
            examples.append(example)
            seen_ids.add(example.id)
            added_hard += 1

    return examples


def summarize_conflict_pair_examples(examples: list[ConflictPairExample]) -> dict[str, Any]:
    """Return small dataset diagnostics for logs, tests, and metadata."""
    by_split: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    split_labels: dict[str, dict[str, Any]] = {}
    dependency_group_splits: dict[str, set[str]] = {}
    text_splits: dict[str, set[str]] = {}
    positives = 0
    for example in examples:
        by_split[example.split] = by_split.get(example.split, 0) + 1
        by_type[example.conflict_type] = by_type.get(example.conflict_type, 0) + 1
        by_source[example.sample_source] = by_source.get(example.sample_source, 0) + 1
        positives += int(example.label == 1)
        split_summary = split_labels.setdefault(
            example.split,
            {"total": 0, "positive": 0, "negative": 0, "dependency_groups": set()},
        )
        split_summary["total"] += 1
        split_summary["positive"] += int(example.label == 1)
        split_summary["negative"] += int(example.label == 0)
        split_summary["dependency_groups"].update({
            example.dependency_group,
            example.paired_dependency_group,
        })
        for group_id in (
            example.dependency_group,
            example.paired_dependency_group,
        ):
            dependency_group_splits.setdefault(group_id, set()).add(example.split)
        for text in (example.text_a.strip(), example.text_b.strip()):
            if text:
                text_splits.setdefault(text, set()).add(example.split)

    return {
        "split_strategy": SPLIT_STRATEGY,
        "total": len(examples),
        "positive": positives,
        "negative": len(examples) - positives,
        "positive_rate": round(positives / len(examples), 4) if examples else 0.0,
        "by_split": dict(sorted(by_split.items())),
        "by_conflict_type": dict(sorted(by_type.items())),
        "by_sample_source": dict(sorted(by_source.items())),
        "by_split_label": {
            split: {
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "dependency_groups"
                },
                "dependency_groups": len(summary["dependency_groups"]),
            }
            for split, summary in sorted(split_labels.items())
        },
        "dependency_group_overlap": sum(
            len(splits) > 1
            for splits in dependency_group_splits.values()
        ),
        "cross_split_text_overlap": sum(
            len(splits) > 1
            for splits in text_splits.values()
        ),
    }


def write_conflict_pair_dataset(
    examples: list[ConflictPairExample],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write split JSONL files plus metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out / "train.jsonl",
        "val": out / "val.jsonl",
        "test": out / "test.jsonl",
    }
    handles = {split: path.open("w", encoding="utf-8") for split, path in paths.items()}
    try:
        for example in examples:
            handles[example.split].write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    file_fingerprints = {
        split: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": sum(1 for example in examples if example.split == split),
        }
        for split, path in paths.items()
    }
    metadata_path = out / "metadata.json"
    metadata = summarize_conflict_pair_examples(examples)
    metadata["schema_version"] = "conflict-pairs-v2"
    metadata["files"] = file_fingerprints
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["metadata"] = metadata_path
    return paths
