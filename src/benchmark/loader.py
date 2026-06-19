"""VeraBench data loader and validator."""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

QUESTION_TYPES = [
    "single_evidence",
    "multi_evidence",
    "conflict",
    "temporal",
    "unanswerable",
    "misleading",
]

VERABENCH_VERSION = "1.1.2"

EVIDENCE_CATEGORIES = [
    "supporting",
    "conflicting",
    "distractor",
    "outdated",
    "partial",
]

CONFLICT_TYPES = [
    "temporal_conflict",
    "numeric_conflict",
    "entity_mismatch",
    "source_disagreement",
    "definitional_conflict",
    "scope_conflict",
]

EXPECTED_BEHAVIORS = [
    "answer_with_citation",
    "answer_with_conflict_note",
    "abstain",
    "correct_premise",
]

DIFFICULTIES = ["easy", "medium", "hard"]

GROUND_TRUTH_STATUSES = ["supported", "refuted", "not_enough_info"]

EXPECTED_BEHAVIOR_BY_TYPE = {
    "single_evidence": "answer_with_citation",
    "multi_evidence": "answer_with_citation",
    "conflict": "answer_with_conflict_note",
    "temporal": "answer_with_citation",
    "unanswerable": "abstain",
    "misleading": "correct_premise",
}


@dataclass
class CorpusDocument:
    doc_id: str
    title: str
    source: str
    date: str | None
    author: str | None
    url: str | None
    content: str
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CorpusDocument":
        return CorpusDocument(
            doc_id=d["doc_id"],
            title=d["title"],
            source=d["source"],
            date=d.get("date"),
            author=d.get("author"),
            url=d.get("url"),
            content=d["content"],
            entities=d.get("entities", []),
            tags=d.get("tags", []),
        )


@dataclass
class EvidenceRef:
    evidence_id: str
    doc_id: str
    text_span: str
    category: str = "supporting"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EvidenceRef":
        return EvidenceRef(
            evidence_id=d["evidence_id"],
            doc_id=d["doc_id"],
            text_span=d["text_span"],
            category=d.get("category", "supporting"),
        )


@dataclass
class ExpectedConflict:
    pair: list[str]
    conflict_type: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExpectedConflict":
        return ExpectedConflict(
            pair=d["pair"],
            conflict_type=d["conflict_type"],
        )


@dataclass
class GroundTruthClaim:
    claim: str
    status: str
    evidence_ids: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GroundTruthClaim":
        return GroundTruthClaim(
            claim=d["claim"],
            status=d["status"],
            evidence_ids=d.get("evidence_ids", []),
        )


@dataclass
class BenchmarkQuestion:
    id: str
    type: str
    question: str
    ground_truth_answer: str
    ground_truth_claims: list[GroundTruthClaim] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    expected_conflicts: list[ExpectedConflict] = field(default_factory=list)
    difficulty: str = "medium"
    requires_multi_hop: bool = False
    expected_behavior: str = "answer_with_citation"
    tags: list[str] = field(default_factory=list)
    annotation_rationale: str | None = None
    difficulty_rationale: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "BenchmarkQuestion":
        return BenchmarkQuestion(
            id=d["id"],
            type=d["type"],
            question=d["question"],
            ground_truth_answer=d["ground_truth_answer"],
            ground_truth_claims=[
                GroundTruthClaim.from_dict(c) for c in d.get("ground_truth_claims", [])
            ],
            evidence=[EvidenceRef.from_dict(e) for e in d.get("evidence", [])],
            expected_conflicts=[
                ExpectedConflict.from_dict(c) for c in d.get("expected_conflicts", [])
            ],
            difficulty=d.get("difficulty", "medium"),
            requires_multi_hop=d.get("requires_multi_hop", False),
            expected_behavior=d.get("expected_behavior", "answer_with_citation"),
            tags=d.get("tags", []),
            annotation_rationale=d.get("annotation_rationale"),
            difficulty_rationale=d.get("difficulty_rationale"),
        )


@dataclass
class VeraBench:
    corpus: dict[str, CorpusDocument]
    questions: list[BenchmarkQuestion]
    version: str = VERABENCH_VERSION

    def get_questions_by_type(self, qtype: str) -> list[BenchmarkQuestion]:
        return [q for q in self.questions if q.type == qtype]

    def get_questions_by_difficulty(self, difficulty: str) -> list[BenchmarkQuestion]:
        return [q for q in self.questions if q.difficulty == difficulty]

    def get_document(self, doc_id: str) -> CorpusDocument | None:
        return self.corpus.get(doc_id)

    def stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        for q in self.questions:
            type_counts[q.type] = type_counts.get(q.type, 0) + 1
        return {
            "total_documents": len(self.corpus),
            "total_questions": len(self.questions),
            "questions_by_type": type_counts,
            "multi_hop_count": sum(1 for q in self.questions if q.requires_multi_hop),
            "conflict_count": sum(1 for q in self.questions if q.expected_conflicts),
        }


def evidence_span_match_kind(text_span: str, document_content: str) -> str:
    """Classify whether an evidence span is reproducibly traceable to a document."""
    if text_span in document_content:
        return "exact"
    segments = [
        segment.strip(" \t\r\n.。…")
        for segment in re.split(r"(?:\.{3}|…+)", text_span)
        if segment.strip(" \t\r\n.。…")
    ]
    if len(segments) < 2:
        return "untraceable"
    cursor = 0
    for segment in segments:
        position = document_content.find(segment, cursor)
        if position < 0:
            return "untraceable"
        cursor = position + len(segment)
    return "segmented"


def evidence_dependency_groups(
    questions: list[BenchmarkQuestion],
) -> dict[str, str]:
    """Group questions connected through shared gold evidence documents."""
    parent = {question.id: question.id for question in questions}

    def find(question_id: str) -> str:
        root = question_id
        while parent[root] != root:
            root = parent[root]
        while parent[question_id] != question_id:
            next_id = parent[question_id]
            parent[question_id] = root
            question_id = next_id
        return root

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        low, high = sorted((first_root, second_root))
        parent[high] = low

    questions_by_document: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        for doc_id in sorted({evidence.doc_id for evidence in question.evidence}):
            questions_by_document[doc_id].append(question.id)
    for question_ids in questions_by_document.values():
        anchor = question_ids[0]
        for question_id in question_ids[1:]:
            union(anchor, question_id)

    components: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        components[find(question.id)].append(question.id)
    ordered = sorted(
        (sorted(question_ids) for question_ids in components.values()),
        key=lambda question_ids: question_ids[0],
    )
    return {
        question_id: f"evidence-component-{index:03d}"
        for index, question_ids in enumerate(ordered, start=1)
        for question_id in question_ids
    }


class VeraBenchLoader:
    DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "verabench"
    PACKAGE_DATA_PATH = resources.files(__package__) / "data" / "verabench"

    def __init__(self, data_dir: str | None = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        elif self.DEFAULT_PATH.exists():
            self.data_dir = self.DEFAULT_PATH
        else:
            self.data_dir = Path(str(self.PACKAGE_DATA_PATH))

    def load(self) -> VeraBench:
        corpus_path = self.data_dir / "corpus.jsonl"
        questions_path = self.data_dir / "questions.jsonl"

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions not found: {questions_path}")

        corpus = {}
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = CorpusDocument.from_dict(json.loads(line))
                    if doc.doc_id in corpus:
                        raise ValueError(
                            f"Benchmark validation failed:\nduplicate corpus id '{doc.doc_id}'"
                        )
                    corpus[doc.doc_id] = doc

        questions = []
        with open(questions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(BenchmarkQuestion.from_dict(json.loads(line)))

        self._validate(corpus, questions)
        return VeraBench(corpus=corpus, questions=questions)

    def _validate(self, corpus: dict[str, CorpusDocument], questions: list[BenchmarkQuestion]):
        errors = []
        seen_question_ids: set[str] = set()
        for q in questions:
            if q.id in seen_question_ids:
                errors.append(f"{q.id}: duplicate question id")
            seen_question_ids.add(q.id)

            if q.type not in QUESTION_TYPES:
                errors.append(f"{q.id}: unknown type '{q.type}'")
            if q.expected_behavior not in EXPECTED_BEHAVIORS:
                errors.append(f"{q.id}: unknown behavior '{q.expected_behavior}'")
            if q.difficulty not in DIFFICULTIES:
                errors.append(f"{q.id}: unknown difficulty '{q.difficulty}'")
            expected_behavior = EXPECTED_BEHAVIOR_BY_TYPE.get(q.type)
            if expected_behavior and q.expected_behavior != expected_behavior:
                errors.append(
                    f"{q.id}: type '{q.type}' requires behavior "
                    f"'{expected_behavior}', got '{q.expected_behavior}'"
                )
            if q.type == "conflict" and not q.expected_conflicts:
                errors.append(f"{q.id}: conflict question has no expected conflicts")
            if q.type != "unanswerable" and not q.evidence:
                errors.append(f"{q.id}: answerable question has no evidence")

            evidence_ids = [e.evidence_id for e in q.evidence]
            if len(evidence_ids) != len(set(evidence_ids)):
                errors.append(f"{q.id}: duplicate evidence ids")
            evidence_id_set = set(evidence_ids)
            for e in q.evidence:
                if e.doc_id not in corpus:
                    errors.append(f"{q.id}: evidence refs missing doc '{e.doc_id}'")
                elif evidence_span_match_kind(
                    e.text_span,
                    corpus[e.doc_id].content,
                ) == "untraceable":
                    errors.append(
                        f"{q.id}: evidence '{e.evidence_id}' text_span is not "
                        f"traceable to document '{e.doc_id}'"
                    )
                if e.category not in EVIDENCE_CATEGORIES:
                    errors.append(f"{q.id}: unknown category '{e.category}'")

            seen_pairs: set[tuple[str, str, str]] = set()
            for c in q.expected_conflicts:
                if c.conflict_type not in CONFLICT_TYPES:
                    errors.append(f"{q.id}: unknown conflict type '{c.conflict_type}'")
                if len(c.pair) != 2:
                    errors.append(f"{q.id}: conflict pair must contain exactly two evidence ids")
                    continue
                missing = [evidence_id for evidence_id in c.pair if evidence_id not in evidence_id_set]
                if missing:
                    errors.append(
                        f"{q.id}: conflict pair references unknown evidence ids {missing}"
                    )
                first_id, second_id = sorted(c.pair)
                pair_key = (first_id, second_id, c.conflict_type)
                if pair_key in seen_pairs:
                    errors.append(
                        f"{q.id}: duplicate conflict pair {c.pair} ({c.conflict_type})"
                    )
                seen_pairs.add(pair_key)

            for claim in q.ground_truth_claims:
                if claim.status not in GROUND_TRUTH_STATUSES:
                    errors.append(
                        f"{q.id}: unknown ground-truth status '{claim.status}'"
                    )
                missing = [
                    evidence_id
                    for evidence_id in claim.evidence_ids
                    if evidence_id not in evidence_id_set
                ]
                if missing:
                    errors.append(
                        f"{q.id}: ground-truth claim references unknown evidence ids {missing}"
                    )
        if errors:
            raise ValueError("Benchmark validation failed:\n" + "\n".join(errors))


def load_verabench(data_dir: str | None = None) -> VeraBench:
    loader = VeraBenchLoader(data_dir)
    return loader.load()
