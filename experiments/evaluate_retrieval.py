#!/usr/bin/env python3
"""Offline VeraBench retrieval evaluation."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.loader import BenchmarkQuestion, VeraBenchLoader  # noqa: E402
from src.evaluation.evidence_metrics import EvidenceMetrics  # noqa: E402
from src.retriever.base import BaseRetriever  # noqa: E402
from src.retriever.bm25 import BM25Retriever  # noqa: E402
from src.retriever.dense import DenseRetriever  # noqa: E402
from src.retriever.hybrid import HybridRetriever  # noqa: E402

RETRIEVERS = ("bm25", "dense", "hybrid")
TOP_K_POLICIES = ("fixed", "precision_cap", "complexity_adaptive")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _documents_for_index(benchmark: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": document.doc_id,
            "title": document.title,
            "text": document.content,
            "source": document.source,
            "date": document.date,
            "author": document.author,
            "url": document.url,
            "entities": document.entities,
            "tags": document.tags,
        }
        for document in benchmark.corpus.values()
    ]


def _make_retriever(
    name: str,
    *,
    dense_model_name: str = "BAAI/bge-base-en-v1.5",
    dense_device: str = "cpu",
    dense_local_files_only: bool = True,
) -> BaseRetriever:
    if name == "bm25":
        return BM25Retriever()
    if name == "dense":
        return DenseRetriever(
            model_name=dense_model_name,
            device=dense_device,
            local_files_only=dense_local_files_only,
        )
    if name == "hybrid":
        return HybridRetriever(
            model_name=dense_model_name,
            device=dense_device,
            local_files_only=dense_local_files_only,
        )
    raise ValueError(f"Unsupported retriever: {name}")


def _ndcg_binary(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 1.0
    dcg = 0.0
    for index, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(index + 2)
    ideal_hits = min(len(relevant), len(retrieved))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    for index, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (index + 1)
    return 0.0


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _select_top_k(
    question: BenchmarkQuestion,
    *,
    retrieval_depth: int,
    policy: str,
) -> int:
    if retrieval_depth < 0:
        raise ValueError("top_k must be non-negative")
    if policy == "fixed":
        return retrieval_depth
    if policy == "precision_cap":
        return min(retrieval_depth, 4)
    if policy == "complexity_adaptive":
        if question.type in {"multi_evidence", "conflict"} or question.requires_multi_hop:
            return min(retrieval_depth, 5)
        if question.type in {"temporal", "misleading"}:
            return min(retrieval_depth, 4)
        return min(retrieval_depth, 2)
    raise ValueError(f"Unsupported top-k policy: {policy}")


def evaluate_questions(
    questions: list[BenchmarkQuestion],
    retriever: BaseRetriever,
    *,
    top_k: int,
    top_k_policy: str = "fixed",
    include_no_gold: bool = False,
) -> list[dict[str, Any]]:
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    rows: list[dict[str, Any]] = []
    for question in questions:
        relevant_doc_ids = sorted({evidence.doc_id for evidence in question.evidence})
        if not relevant_doc_ids and not include_no_gold:
            continue
        retrieval_depth = top_k
        selected_top_k = _select_top_k(
            question,
            retrieval_depth=retrieval_depth,
            policy=top_k_policy,
        )
        results = retriever.retrieve(question.question, top_k=retrieval_depth)
        retrieved_doc_ids = _unique_preserving_order(
            [result.doc_id for result in results]
        )[:selected_top_k]
        relevant_set = set(relevant_doc_ids)
        hit_count = len(set(retrieved_doc_ids) & relevant_set)
        precision = EvidenceMetrics.evidence_precision(
            retrieved_doc_ids,
            relevant_doc_ids,
        )
        recall = EvidenceMetrics.evidence_recall(
            retrieved_doc_ids,
            relevant_doc_ids,
        )
        f1 = EvidenceMetrics.evidence_f1(retrieved_doc_ids, relevant_doc_ids)
        rows.append({
            "question_id": question.id,
            "question_type": question.type,
            "difficulty": question.difficulty,
            "requires_multi_hop": question.requires_multi_hop,
            "question": question.question,
            "gold_document_ids": relevant_doc_ids,
            "retrieved_document_ids": retrieved_doc_ids,
            "retrieval_depth": retrieval_depth,
            "selected_top_k": selected_top_k,
            "top_k_policy": top_k_policy,
            "hit_count": hit_count,
            "retrieved_count": len(retrieved_doc_ids),
            "gold_count": len(relevant_doc_ids),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "hit": hit_count > 0 or not relevant_doc_ids,
            "all_gold_retrieved": recall >= 1.0,
            "mrr": round(_mrr(retrieved_doc_ids, relevant_set), 6),
            "ndcg": round(_ndcg_binary(retrieved_doc_ids, relevant_set), 6),
        })
    return rows


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[field]) for row in rows) / len(rows)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "questions": 0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "hit_rate": 0.0,
            "all_gold_retrieved_rate": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
        }
    total_hits = sum(int(row["hit_count"]) for row in rows)
    total_retrieved = sum(int(row["retrieved_count"]) for row in rows)
    total_gold = sum(int(row["gold_count"]) for row in rows)
    return {
        "questions": len(rows),
        "macro_precision": round(_mean(rows, "precision"), 6),
        "macro_recall": round(_mean(rows, "recall"), 6),
        "macro_f1": round(_mean(rows, "f1"), 6),
        "hit_rate": round(sum(bool(row["hit"]) for row in rows) / len(rows), 6),
        "all_gold_retrieved_rate": round(
            sum(bool(row["all_gold_retrieved"]) for row in rows) / len(rows),
            6,
        ),
        "mrr": round(_mean(rows, "mrr"), 6),
        "ndcg": round(_mean(rows, "ndcg"), 6),
        "micro_precision": round(total_hits / total_retrieved, 6) if total_retrieved else 0.0,
        "micro_recall": round(total_hits / total_gold, 6) if total_gold else 1.0,
    }


def _grouped(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: _aggregate(value) for key, value in sorted(groups.items())}


def build_report(
    *,
    data_dir: str | None,
    retriever_name: str,
    top_k: int,
    top_k_policy: str = "fixed",
    sweep_top_k: list[int] | None = None,
    question_types: list[str] | None = None,
    question_ids: list[str] | None = None,
    max_questions: int | None = None,
    include_no_gold: bool = False,
    dense_model_name: str = "BAAI/bge-base-en-v1.5",
    dense_device: str = "cpu",
    dense_local_files_only: bool = True,
) -> dict[str, Any]:
    loader = VeraBenchLoader(data_dir)
    benchmark = loader.load()
    documents = _documents_for_index(benchmark)
    retriever = _make_retriever(
        retriever_name,
        dense_model_name=dense_model_name,
        dense_device=dense_device,
        dense_local_files_only=dense_local_files_only,
    )
    retriever.index_documents(documents)

    questions = benchmark.questions
    if question_types:
        requested_types = set(question_types)
        questions = [question for question in questions if question.type in requested_types]
    if question_ids:
        requested_ids = set(question_ids)
        known_ids = {question.id for question in benchmark.questions}
        unknown_ids = sorted(requested_ids - known_ids)
        if unknown_ids:
            raise ValueError(f"Unknown VeraBench question id(s): {', '.join(unknown_ids)}")
        questions = [question for question in questions if question.id in requested_ids]
    if max_questions is not None:
        questions = questions[:max_questions]

    rows = evaluate_questions(
        questions,
        retriever,
        top_k=top_k,
        top_k_policy=top_k_policy,
        include_no_gold=include_no_gold,
    )
    sweep = []
    for sweep_k in sweep_top_k or []:
        if sweep_k == top_k:
            sweep_rows = rows
        else:
            sweep_rows = evaluate_questions(
                questions,
                retriever,
                top_k=sweep_k,
                top_k_policy=top_k_policy,
                include_no_gold=include_no_gold,
            )
        sweep.append({
            "top_k": sweep_k,
            "top_k_policy": top_k_policy,
            "summary": _aggregate(sweep_rows),
        })
    return {
        "schema_version": "retrieval-eval-v1",
        "retriever": retriever_name,
        "top_k": top_k,
        "top_k_policy": top_k_policy,
        "dense_model_name": dense_model_name if retriever_name in {"dense", "hybrid"} else "",
        "dense_local_files_only": (
            dense_local_files_only if retriever_name in {"dense", "hybrid"} else None
        ),
        "include_no_gold": include_no_gold,
        "benchmark": {
            "version": benchmark.version,
            "questions": len(benchmark.questions),
            "documents": len(benchmark.corpus),
            "fingerprints": {
                "corpus_sha256": _sha256(loader.data_dir / "corpus.jsonl"),
                "questions_sha256": _sha256(loader.data_dir / "questions.jsonl"),
            },
        },
        "selected_questions": len(questions),
        "evaluated_questions": len(rows),
        "summary": _aggregate(rows),
        "by_type": _grouped(rows, "question_type"),
        "by_difficulty": _grouped(rows, "difficulty"),
        "by_multi_hop": _grouped(rows, "requires_multi_hop"),
        "sweep": sweep,
        "question_results": rows,
    }


def _select_questions(
    questions: list[BenchmarkQuestion],
    *,
    question_types: list[str] | None = None,
    question_ids: list[str] | None = None,
    max_questions: int | None = None,
) -> list[BenchmarkQuestion]:
    selected = questions
    if question_types:
        requested_types = set(question_types)
        selected = [
            question for question in selected
            if question.type in requested_types
        ]
    if question_ids:
        requested_ids = set(question_ids)
        known_ids = {question.id for question in questions}
        unknown_ids = sorted(requested_ids - known_ids)
        if unknown_ids:
            raise ValueError(
                f"Unknown VeraBench question id(s): {', '.join(unknown_ids)}"
            )
        selected = [question for question in selected if question.id in requested_ids]
    if max_questions is not None:
        selected = selected[:max_questions]
    return selected


def build_matrix_report(
    *,
    data_dir: str | None,
    retriever_names: list[str],
    top_k_values: list[int],
    top_k_policies: list[str],
    question_types: list[str] | None = None,
    question_ids: list[str] | None = None,
    max_questions: int | None = None,
    include_no_gold: bool = False,
    dense_model_name: str = "BAAI/bge-base-en-v1.5",
    dense_device: str = "cpu",
    dense_local_files_only: bool = True,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    loader = VeraBenchLoader(data_dir)
    benchmark = loader.load()
    documents = _documents_for_index(benchmark)
    questions = _select_questions(
        benchmark.questions,
        question_types=question_types,
        question_ids=question_ids,
        max_questions=max_questions,
    )
    variants: list[dict[str, Any]] = []
    for retriever_name in retriever_names:
        try:
            retriever = _make_retriever(
                retriever_name,
                dense_model_name=dense_model_name,
                dense_device=dense_device,
                dense_local_files_only=dense_local_files_only,
            )
            retriever.index_documents(documents)
        except Exception as exc:
            if not continue_on_error:
                raise
            for top_k in top_k_values:
                for top_k_policy in top_k_policies:
                    variants.append({
                        "retriever": retriever_name,
                        "top_k": top_k,
                        "top_k_policy": top_k_policy,
                        "status": "error",
                        "error": str(exc),
                    })
            continue

        for top_k in top_k_values:
            for top_k_policy in top_k_policies:
                try:
                    rows = evaluate_questions(
                        questions,
                        retriever,
                        top_k=top_k,
                        top_k_policy=top_k_policy,
                        include_no_gold=include_no_gold,
                    )
                    variants.append({
                        "retriever": retriever_name,
                        "top_k": top_k,
                        "top_k_policy": top_k_policy,
                        "status": "ok",
                        "evaluated_questions": len(rows),
                        "summary": _aggregate(rows),
                        "by_type": _grouped(rows, "question_type"),
                    })
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    variants.append({
                        "retriever": retriever_name,
                        "top_k": top_k,
                        "top_k_policy": top_k_policy,
                        "status": "error",
                        "error": str(exc),
                    })

    successful = [variant for variant in variants if variant["status"] == "ok"]
    best_by_macro_f1 = None
    if successful:
        best_by_macro_f1 = max(
            successful,
            key=lambda variant: (
                float(variant["summary"]["macro_f1"]),
                float(variant["summary"]["macro_precision"]),
                float(variant["summary"]["macro_recall"]),
            ),
        )

    return {
        "schema_version": "retrieval-matrix-v1",
        "retrievers": retriever_names,
        "top_k_values": top_k_values,
        "top_k_policies": top_k_policies,
        "dense_model_name": dense_model_name,
        "dense_local_files_only": dense_local_files_only,
        "include_no_gold": include_no_gold,
        "benchmark": {
            "version": benchmark.version,
            "questions": len(benchmark.questions),
            "documents": len(benchmark.corpus),
            "fingerprints": {
                "corpus_sha256": _sha256(loader.data_dir / "corpus.jsonl"),
                "questions_sha256": _sha256(loader.data_dir / "questions.jsonl"),
            },
        },
        "selected_questions": len(questions),
        "variants": variants,
        "best_by_macro_f1": best_by_macro_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VeraBench retrieval offline")
    parser.add_argument("--data-dir", help="Optional VeraBench data directory")
    parser.add_argument(
        "--retriever",
        choices=RETRIEVERS,
        default="bm25",
        help=(
            "Retriever variant to evaluate. Hybrid falls back to BM25 if dense "
            "is unavailable."
        ),
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run a retriever/top-k/policy matrix instead of one detailed report.",
    )
    parser.add_argument(
        "--matrix-retrievers",
        nargs="+",
        choices=RETRIEVERS,
        default=["bm25"],
        help="Retriever variants to include with --matrix.",
    )
    parser.add_argument(
        "--matrix-top-k",
        nargs="+",
        type=int,
        default=[5, 10],
        help="Retrieval depths to include with --matrix.",
    )
    parser.add_argument(
        "--matrix-policies",
        nargs="+",
        choices=TOP_K_POLICIES,
        default=["fixed", "precision_cap", "complexity_adaptive"],
        help="Top-k selection policies to include with --matrix.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="In --matrix mode, stop on the first retriever/indexing error.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--dense-model-name",
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer model for dense/hybrid retrieval.",
    )
    parser.add_argument(
        "--dense-device",
        default="cpu",
        help="Device for dense/hybrid SentenceTransformer retrieval.",
    )
    parser.add_argument(
        "--dense-allow-download",
        action="store_true",
        help=(
            "Allow dense/hybrid retrieval to download missing models. By default "
            "offline evaluation uses local cached model files only."
        ),
    )
    parser.add_argument(
        "--top-k-policy",
        choices=TOP_K_POLICIES,
        default="fixed",
        help=(
            "Post-retrieval selection policy. fixed keeps --top-k; "
            "precision_cap caps at 4; complexity_adaptive uses smaller caps "
            "for simple rows and larger caps for multi-hop/conflict rows."
        ),
    )
    parser.add_argument(
        "--sweep-top-k",
        nargs="+",
        type=int,
        help="Also report summaries for these retrieval depths under the selected policy.",
    )
    parser.add_argument("--types", nargs="+", help="Question types to include")
    parser.add_argument("--ids", nargs="+", help="Question ids to include")
    parser.add_argument("--max", type=int, dest="max_questions", help="Limit selected questions")
    parser.add_argument(
        "--include-no-gold",
        action="store_true",
        help="Include questions with no gold evidence, such as unanswerable rows.",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    if args.matrix:
        report = build_matrix_report(
            data_dir=args.data_dir,
            retriever_names=args.matrix_retrievers,
            top_k_values=args.matrix_top_k,
            top_k_policies=args.matrix_policies,
            question_types=args.types,
            question_ids=args.ids,
            max_questions=args.max_questions,
            include_no_gold=args.include_no_gold,
            dense_model_name=args.dense_model_name,
            dense_device=args.dense_device,
            dense_local_files_only=not args.dense_allow_download,
            continue_on_error=not args.fail_fast,
        )
    else:
        report = build_report(
            data_dir=args.data_dir,
            retriever_name=args.retriever,
            top_k=args.top_k,
            top_k_policy=args.top_k_policy,
            sweep_top_k=args.sweep_top_k,
            question_types=args.types,
            question_ids=args.ids,
            max_questions=args.max_questions,
            include_no_gold=args.include_no_gold,
            dense_model_name=args.dense_model_name,
            dense_device=args.dense_device,
            dense_local_files_only=not args.dense_allow_download,
        )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
