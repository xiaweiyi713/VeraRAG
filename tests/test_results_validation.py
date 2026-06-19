"""Tests for published VeraBench results validation."""

import json
from pathlib import Path

from experiments.validate_results import main, validate_results


def test_results_validation_accepts_repository_results():
    audit = validate_results()

    assert audit.valid
    assert audit.errors == []
    assert audit.corpus_sha256
    assert audit.questions_sha256


def test_results_validation_reports_stale_results_page(tmp_path):
    _write_results_fixture(
        tmp_path,
        results=(
            "# VeraBench Results\n\n"
            "Generation command:\n\n"
            "```bash\n"
            "python experiments/build_verabench_leaderboard.py results/full.json --output docs/RESULTS.md\n"
            "```\n\n"
            "Current repository data is VeraBench v1.1.2\n"
        ),
    )

    audit = validate_results(tmp_path)

    assert not audit.valid
    assert (
        "docs/RESULTS.md generation command missing --allow-unverified"
        in audit.errors
    )
    assert (
        "docs/RESULTS.md generation command missing --allow-mixed-benchmarks"
        in audit.errors
    )
    assert any("missing current Corpus SHA-256" in error for error in audit.errors)
    assert any("missing required table header" in error for error in audit.errors)


def test_results_validation_requires_formal_leaderboard_integrity_docs(tmp_path):
    docs = "Formal leaderboard generation rejects demo reports only.\n"
    _write_results_fixture(tmp_path, evaluation=docs, card="")

    audit = validate_results(tmp_path)

    assert not audit.valid
    assert (
        "leaderboard documentation missing integrity rule: mixed benchmark fingerprints"
        in audit.errors
    )
    assert (
        "leaderboard documentation missing integrity rule: --allow-unverified"
        in audit.errors
    )


def test_results_validation_cli_json(capsys):
    exit_code = main(["--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["docs_results_path"].endswith("docs/RESULTS.md")


def _write_results_fixture(
    root: Path,
    *,
    results: str | None = None,
    evaluation: str | None = None,
    card: str | None = None,
) -> None:
    (root / "docs").mkdir()
    (root / "data" / "verabench").mkdir(parents=True)
    corpus = root / "data" / "verabench" / "corpus.jsonl"
    questions = root / "data" / "verabench" / "questions.jsonl"
    corpus.write_text('{"id":"doc-1","content":"Alpha"}\n', encoding="utf-8")
    questions.write_text('{"id":"V001","question":"Alpha?"}\n', encoding="utf-8")

    corpus_sha = _sha256(corpus)
    questions_sha = _sha256(questions)
    valid_results = (
        "# VeraBench Results\n\n"
        "> v1.0 and v1.1 rows are historical and must not be compared directly.\n\n"
        "Generation command:\n\n"
        "```bash\n"
        "python experiments/build_verabench_leaderboard.py results/full.json "
        "--allow-unverified --allow-mixed-benchmarks --output docs/RESULTS.md\n"
        "```\n\n"
        "legacy, non-comparable historical runs\n\n"
        "## VeraBench v1.1.2 Status\n\n"
        "Current repository data is VeraBench v1.1.2\n\n"
        f"Corpus SHA-256: `{corpus_sha}`\n\n"
        f"Questions SHA-256: `{questions_sha}`\n\n"
        "A v1.1.2 targeted rerun should use a current config.\n\n"
        "## Historical VeraBench v1.1 Full Run\n\n"
        "| Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict micro-F1 |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n\n"
        "## Historical v1.0 Leaderboard\n\n"
        "| Rank | Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict F1 | ECE | Avg Latency | Commit |\n"
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n\n"
        "## Focused Conflict Smoke\n\n"
        "This v1.0 focused smoke is not a full leaderboard entry.\n\n"
        "## Reproducibility Metadata\n\n"
        "| Run | Provider | Model | Config | Timestamp | Result Path |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    valid_docs = (
        "Formal leaderboard generation rejects demo reports, incomplete or errored "
        "runs, missing reproducibility metadata, and mixed benchmark fingerprints. "
        "Use --allow-unverified or --allow-mixed-benchmarks only for historical "
        "diagnostics."
    )
    (root / "docs" / "RESULTS.md").write_text(
        results if results is not None else valid_results,
        encoding="utf-8",
    )
    (root / "docs" / "EVALUATION.md").write_text(
        evaluation if evaluation is not None else valid_docs,
        encoding="utf-8",
    )
    (root / "docs" / "VERABENCH_CARD.md").write_text(
        card if card is not None else valid_docs,
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
