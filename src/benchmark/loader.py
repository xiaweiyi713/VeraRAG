"""VeraBench data loader and validator."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


QUESTION_TYPES = [
    "single_evidence",
    "multi_evidence",
    "conflict",
    "temporal",
    "unanswerable",
    "misleading",
]

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


@dataclass
class CorpusDocument:
    doc_id: str
    title: str
    source: str
    date: Optional[str]
    author: Optional[str]
    url: Optional[str]
    content: str
    entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CorpusDocument":
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
    def from_dict(d: Dict[str, Any]) -> "EvidenceRef":
        return EvidenceRef(
            evidence_id=d["evidence_id"],
            doc_id=d["doc_id"],
            text_span=d["text_span"],
            category=d.get("category", "supporting"),
        )


@dataclass
class ExpectedConflict:
    pair: List[str]
    conflict_type: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExpectedConflict":
        return ExpectedConflict(
            pair=d["pair"],
            conflict_type=d["conflict_type"],
        )


@dataclass
class GroundTruthClaim:
    claim: str
    status: str
    evidence_ids: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GroundTruthClaim":
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
    ground_truth_claims: List[GroundTruthClaim] = field(default_factory=list)
    evidence: List[EvidenceRef] = field(default_factory=list)
    expected_conflicts: List[ExpectedConflict] = field(default_factory=list)
    difficulty: str = "medium"
    requires_multi_hop: bool = False
    expected_behavior: str = "answer_with_citation"
    tags: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BenchmarkQuestion":
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
        )


@dataclass
class VeraBench:
    corpus: Dict[str, CorpusDocument]
    questions: List[BenchmarkQuestion]
    version: str = "1.0"

    def get_questions_by_type(self, qtype: str) -> List[BenchmarkQuestion]:
        return [q for q in self.questions if q.type == qtype]

    def get_questions_by_difficulty(self, difficulty: str) -> List[BenchmarkQuestion]:
        return [q for q in self.questions if q.difficulty == difficulty]

    def get_document(self, doc_id: str) -> Optional[CorpusDocument]:
        return self.corpus.get(doc_id)

    def stats(self) -> Dict[str, Any]:
        type_counts = {}
        for q in self.questions:
            type_counts[q.type] = type_counts.get(q.type, 0) + 1
        return {
            "total_documents": len(self.corpus),
            "total_questions": len(self.questions),
            "questions_by_type": type_counts,
            "multi_hop_count": sum(1 for q in self.questions if q.requires_multi_hop),
            "conflict_count": sum(1 for q in self.questions if q.expected_conflicts),
        }


class VeraBenchLoader:
    DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "verabench"

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_PATH

    def load(self) -> VeraBench:
        corpus_path = self.data_dir / "corpus.jsonl"
        questions_path = self.data_dir / "questions.jsonl"

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions not found: {questions_path}")

        corpus = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = CorpusDocument.from_dict(json.loads(line))
                    corpus[doc.doc_id] = doc

        questions = []
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(BenchmarkQuestion.from_dict(json.loads(line)))

        self._validate(corpus, questions)
        return VeraBench(corpus=corpus, questions=questions)

    def _validate(self, corpus: Dict[str, CorpusDocument], questions: List[BenchmarkQuestion]):
        errors = []
        for q in questions:
            if q.type not in QUESTION_TYPES:
                errors.append(f"{q.id}: unknown type '{q.type}'")
            if q.expected_behavior not in EXPECTED_BEHAVIORS:
                errors.append(f"{q.id}: unknown behavior '{q.expected_behavior}'")
            for e in q.evidence:
                if e.doc_id not in corpus:
                    errors.append(f"{q.id}: evidence refs missing doc '{e.doc_id}'")
                if e.category not in EVIDENCE_CATEGORIES:
                    errors.append(f"{q.id}: unknown category '{e.category}'")
            for c in q.expected_conflicts:
                if c.conflict_type not in CONFLICT_TYPES:
                    errors.append(f"{q.id}: unknown conflict type '{c.conflict_type}'")
        if errors:
            raise ValueError(f"Benchmark validation failed:\n" + "\n".join(errors))


def load_verabench(data_dir: Optional[str] = None) -> VeraBench:
    loader = VeraBenchLoader(data_dir)
    return loader.load()
