"""Tests for VeraBench conflict-pair training data builders."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from experiments.run_conflict_ablation import (
    compare_summaries,
    summarize_report,
)
from experiments.train_conflict_cross_encoder import (
    _balance_training_rows,
    _best_threshold,
    _classification_metrics,
    _cross_encoder_init_kwargs,
    _pair_dependency_components,
    _prediction_rows,
    _save_trained_model,
    _score_to_probability,
    _validate_split_integrity,
)
from src.benchmark.conflict_pairs import (
    SPLIT_STRATEGY,
    _dependency_group_split_maps,
    build_conflict_pair_examples,
    summarize_conflict_pair_examples,
    write_conflict_pair_dataset,
)
from src.benchmark.loader import (
    VeraBenchLoader,
    evidence_dependency_groups,
    load_verabench,
)
from tests.test_benchmark import SAMPLE_CORPUS, SAMPLE_QUESTIONS


@pytest.fixture
def temp_bench_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for doc in SAMPLE_CORPUS:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        with open(os.path.join(tmpdir, "questions.jsonl"), "w", encoding="utf-8") as f:
            for question in SAMPLE_QUESTIONS:
                f.write(json.dumps(question, ensure_ascii=False) + "\n")
        yield tmpdir


def test_build_conflict_pair_examples_from_sample(temp_bench_dir):
    bench = VeraBenchLoader(temp_bench_dir).load()

    examples = build_conflict_pair_examples(benchmark=bench, max_negative_per_question=2)
    summary = summarize_conflict_pair_examples(examples)

    assert summary["total"] >= 1
    assert summary["positive"] == 1
    assert summary["by_conflict_type"]["numeric_conflict"] == 1
    positive = next(example for example in examples if example.label == 1)
    assert positive.question_id == "T002"
    assert positive.text_a
    assert positive.text_b
    assert positive.split in {"train", "val", "test"}


def test_write_conflict_pair_dataset(tmp_path, temp_bench_dir):
    bench = VeraBenchLoader(temp_bench_dir).load()
    examples = build_conflict_pair_examples(benchmark=bench)

    paths = write_conflict_pair_dataset(examples, tmp_path)

    assert paths["train"].exists()
    assert paths["val"].exists()
    assert paths["test"].exists()
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["positive"] == 1
    assert metadata["schema_version"] == "conflict-pairs-v2"
    assert len(metadata["files"]["train"]["sha256"]) == 64


def test_default_verabench_conflict_pair_summary():
    examples = build_conflict_pair_examples(max_negative_per_question=4)
    summary = summarize_conflict_pair_examples(examples)

    assert summary["total"] > 25
    assert summary["positive"] >= 28
    assert summary["negative"] > 0
    assert "numeric_conflict" in summary["by_conflict_type"]
    assert summary["by_sample_source"]["gold_self_conflict"] >= 1
    assert summary["by_sample_source"]["hard_negative"] >= 1
    assert summary["split_strategy"] == SPLIT_STRATEGY
    assert summary["dependency_group_overlap"] == 0
    assert summary["cross_split_text_overlap"] == 0
    assert summary["by_split_label"]["val"]["positive"] > 0
    assert summary["by_split_label"]["test"]["positive"] > 0


def test_dependency_groups_are_kept_in_one_stratified_split():
    benchmark = load_verabench()
    split_by_question, split_by_group = _dependency_group_split_maps(
        benchmark.questions,
        0.8,
        0.1,
    )
    dependency_group_by_question = evidence_dependency_groups(
        benchmark.questions,
    )

    assert set(split_by_group.values()) == {"train", "val", "test"}
    assert {
        split_by_question[question.id]
        for question in benchmark.questions
        if question.expected_conflicts
    } == {"train", "val", "test"}
    for question in benchmark.questions:
        group_id = dependency_group_by_question[question.id]
        assert split_by_question[question.id] == split_by_group[group_id]


@pytest.mark.parametrize(
    ("train_ratio", "val_ratio"),
    [(0.0, 0.1), (0.8, -0.1), (0.9, 0.1), (1.0, 0.0)],
)
def test_dependency_split_rejects_invalid_ratios(train_ratio, val_ratio):
    benchmark = load_verabench()

    with pytest.raises(ValueError, match="train_ratio"):
        _dependency_group_split_maps(
            benchmark.questions,
            train_ratio,
            val_ratio,
        )


def test_gold_self_conflicts_are_included():
    examples = build_conflict_pair_examples(
        max_negative_per_question=0,
        max_hard_negative_per_question=0,
        max_weak_positive_per_question=0,
    )

    self_conflicts = [
        example for example in examples
        if example.sample_source == "gold_self_conflict"
    ]
    assert self_conflicts
    assert all(example.label == 1 for example in self_conflicts)
    assert all(example.evidence_id_a == example.evidence_id_b for example in self_conflicts)
    assert any(example.text_a != example.text_b for example in self_conflicts)


def test_hard_negatives_are_topical_cross_question_pairs():
    examples = build_conflict_pair_examples(
        max_negative_per_question=0,
        max_hard_negative_per_question=2,
        max_weak_positive_per_question=0,
    )

    hard_negatives = [
        example for example in examples
        if example.sample_source == "hard_negative"
    ]
    assert hard_negatives
    assert all(example.label == 0 for example in hard_negatives)
    assert all(":hard:" in example.id for example in hard_negatives)
    assert all(example.doc_id_a != example.doc_id_b for example in hard_negatives)
    assert all(
        example.dependency_group != example.paired_dependency_group
        or example.question_id != example.paired_question_id
        for example in hard_negatives
    )


def test_build_conflict_training_data_cli(tmp_path):
    output_dir = tmp_path / "pairs"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/build_conflict_training_data.py",
            "--output-dir",
            str(output_dir),
            "--max-negative-per-question",
            "2",
            "--max-hard-negative-per-question",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["summary"]["positive"] >= 28
    assert payload["summary"]["by_sample_source"]["hard_negative"] >= 1
    assert payload["summary"]["benchmark"]["version"] == "1.1.2"
    assert len(
        payload["summary"]["benchmark"]["fingerprints"]["questions_sha256"]
    ) == 64
    assert len(payload["summary"]["files"]["train"]["sha256"]) == 64
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "metadata.json").exists()


def test_compare_conflict_detectors_cli_rules_only(tmp_path, temp_bench_dir):
    output_path = tmp_path / "conflict_ablation.json"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/compare_conflict_detectors.py",
            "--data-dir",
            temp_bench_dir,
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert output_path.exists()
    assert payload["variants"][0]["name"] == "rules"
    assert payload["variants"][0]["summary"]["gold_conflicts"] == 1
    assert "f1" in payload["variants"][0]["summary"]
    diagnosis = payload["diagnosis"]["variants"]["rules"]
    assert diagnosis["dominant_failure"] in {"none", "under_detection", "mixed"}
    assert "actionable_next_step" in diagnosis


def test_compare_conflict_detectors_can_limit_dependency_split(tmp_path):
    output_path = tmp_path / "conflict_test_split.json"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/compare_conflict_detectors.py",
            "--split",
            "test",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["evaluation_scope"]["split"] == "test"
    assert payload["evaluation_scope"]["split_strategy"] == SPLIT_STRATEGY
    assert payload["variants"][0]["summary"]["questions"] > 0
    assert payload["variants"][0]["summary"]["questions"] < 13
    assert payload["diagnosis"]["variants"]["rules"]["gold_conflicts"] > 0


def test_compare_conflict_detectors_marks_external_independent_test(
    tmp_path,
    temp_bench_dir,
):
    output_path = tmp_path / "external_conflicts.json"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/compare_conflict_detectors.py",
            "--data-dir",
            temp_bench_dir,
            "--independent-test",
            "--evaluation-id",
            "sample-external-v1",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    scope = json.loads(result.stdout)["evaluation_scope"]
    assert scope["independent_test"] is True
    assert scope["evaluation_id"] == "sample-external-v1"
    assert scope["dataset"]["source"] == "external-data-dir"
    assert len(scope["dataset"]["fingerprints"]["questions_sha256"]) == 64


def test_compare_conflict_detectors_rejects_invalid_split_ratios():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/compare_conflict_detectors.py",
            "--split",
            "test",
            "--train-ratio",
            "0.9",
            "--val-ratio",
            "0.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "train_ratio + val_ratio < 1" in result.stderr


def test_conflict_detector_comparison_diagnosis_reports_learned_effect():
    from experiments.compare_conflict_detectors import _comparison_diagnosis

    payload = _comparison_diagnosis([
        {
            "name": "rules",
            "summary": {
                "gold_conflicts": 10,
                "predicted_conflicts": 2,
                "true_positives": 2,
                "false_positives": 0,
                "false_negatives": 8,
                "precision": 1.0,
                "recall": 0.2,
                "f1": 0.3333,
            },
            "questions": [
                {
                    "question_id": "V001",
                    "question_type": "conflict",
                    "gold": 2,
                    "predicted": 0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 2,
                    "predicted_pairs": [],
                    "gold_pairs": [["E1", "E2"], ["E3", "E4"]],
                }
            ],
        },
        {
            "name": "rules_plus_learned",
            "summary": {
                "gold_conflicts": 10,
                "predicted_conflicts": 6,
                "true_positives": 6,
                "false_positives": 0,
                "false_negatives": 4,
                "precision": 1.0,
                "recall": 0.6,
                "f1": 0.75,
            },
            "questions": [],
        },
    ])

    rules = payload["variants"]["rules"]
    assert rules["dominant_failure"] == "under_detection"
    assert rules["top_false_negative_questions"][0]["question_id"] == "V001"
    assert payload["comparison"]["recall_delta"] == 0.4
    assert payload["comparison"]["false_positive_delta"] == 0
    assert payload["comparison"]["learned_effect"] == "promising_recall_gain_without_extra_fp"


def test_conflict_ablation_plan_only_cli(tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
llm:
  provider: deepseek
  model: deepseek-chat
pipeline:
  enable_conflict_graph: true
conflict_graph:
  enable_learned_detector: false
  learned_threshold: 0.7
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "ablation"

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_conflict_ablation.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--learned-model-path",
            "models/conflict",
            "--types",
            "conflict",
            "--max",
            "2",
            "--plan-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout.split("\n\nSummary saved to ")[0])
    assert payload["plan"]["mode"] == "plan"
    assert (output_dir / "configs" / "rules.yaml").exists()
    assert (output_dir / "configs" / "rules_plus_learned.yaml").exists()
    assert payload["plan"]["runs"][0]["command"][:3] == [
        sys.executable,
        "-m",
        "experiments.run_verabench",
    ]
    assert "--max" in payload["plan"]["runs"][0]["command"]


def test_conflict_ablation_summary_delta():
    baseline = summarize_report({
        "overall_answer_f1": 0.4,
        "overall_conflict_f1": 0.5,
        "conflict_summary": {
            "predicted_conflicts": 5,
            "true_positives": 3,
            "false_positives": 2,
            "false_negatives": 4,
            "precision": 0.6,
            "recall": 0.428571,
            "f1": 0.5,
        },
    })
    learned = summarize_report({
        "overall_answer_f1": 0.45,
        "overall_conflict_f1": 0.75,
        "conflict_summary": {
            "predicted_conflicts": 7,
            "true_positives": 6,
            "false_positives": 1,
            "false_negatives": 1,
            "precision": 0.857143,
            "recall": 0.857143,
            "f1": 0.857143,
        },
    })

    delta = compare_summaries(baseline, learned)

    assert delta["overall_answer_f1"] == 0.05
    assert delta["overall_conflict_f1"] == 0.25
    assert delta["conflict_summary"]["true_positives"] == 3
    assert delta["conflict_summary"]["false_negatives"] == -3


def test_train_conflict_cross_encoder_dry_run(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    row = {
        "text_a": SAMPLE_CORPUS[0]["content"],
        "text_b": "train-only",
        "label": 1,
        "split": "train",
        "question_id": "Q-train",
        "paired_question_id": "Q-train",
        "dependency_group": "G-train",
        "paired_dependency_group": "G-train",
    }
    val_row = {
        **row,
        "text_a": SAMPLE_CORPUS[1]["content"],
        "text_b": "val-only",
        "split": "val",
        "question_id": "Q-val",
        "paired_question_id": "Q-val",
        "dependency_group": "G-val",
        "paired_dependency_group": "G-val",
    }
    test_row = {
        **row,
        "text_a": "test evidence",
        "text_b": "test-only",
        "split": "test",
        "question_id": "Q-test",
        "paired_question_id": "Q-test",
        "dependency_group": "G-test",
        "paired_dependency_group": "G-test",
    }
    train_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(val_row, ensure_ascii=False) + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(test_row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "experiments/train_conflict_cross_encoder.py",
            "--train",
            str(train_path),
            "--val",
            str(val_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["train"]["total"] == 1
    assert payload["train_loader"]["total"] == 1
    assert payload["train"]["positive"] == 1
    assert payload["test"]["total"] == 1
    assert payload["split_integrity"]["status"] == "verified"


def test_train_conflict_cli_runs_outside_source_directory(tmp_path):
    script = (
        os.path.dirname(os.path.dirname(__file__))
        + "/experiments/train_conflict_cross_encoder.py"
    )
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Train VeraBench conflict CrossEncoder" in result.stdout


def test_training_split_integrity_rejects_dependency_leakage():
    shared = {
        "question_id": "Q1",
        "paired_question_id": "Q1",
        "dependency_group": "G1",
        "paired_dependency_group": "G1",
        "text_a": "shared evidence",
        "text_b": "other evidence",
    }
    rows = {
        "train": [{**shared, "split": "train"}],
        "val": [{**shared, "split": "val"}],
        "test": [],
    }

    with pytest.raises(ValueError, match="cross-split leakage"):
        _validate_split_integrity(rows)


def test_cross_encoder_init_kwargs_reinitialize_binary_head():
    kwargs = _cross_encoder_init_kwargs("cuda")

    assert kwargs["num_labels"] == 1
    assert kwargs["device"] == "cuda"
    assert kwargs["model_kwargs"]["ignore_mismatched_sizes"] is True
    assert kwargs["config_kwargs"]["id2label"] == {0: "CONFLICT"}


def test_training_save_disables_remote_model_card_generation(tmp_path):
    class RecordingModel:
        def __init__(self):
            self.calls = []

        def save(self, path, **kwargs):
            self.calls.append((path, kwargs))

    model = RecordingModel()

    _save_trained_model(model, tmp_path / "model")

    assert model.calls == [
        (str(tmp_path / "model"), {"create_model_card": False}),
    ]


def test_prediction_rows_connect_cross_group_hard_negatives():
    rows = [
        {
            "id": "p1",
            "question_id": "Q1",
            "paired_question_id": "Q1",
            "dependency_group": "G1",
            "paired_dependency_group": "G1",
            "label": 1,
        },
        {
            "id": "n1",
            "question_id": "Q1",
            "paired_question_id": "Q2",
            "dependency_group": "G1",
            "paired_dependency_group": "G2",
            "label": 0,
        },
    ]

    components = _pair_dependency_components(rows)
    predictions = _prediction_rows(rows, [0.9, 0.2], 0.5)

    assert components["G1"] == components["G2"]
    assert {
        row["evaluation_dependency_component"] for row in predictions
    } == {components["G1"]}
    assert [row["predicted"] for row in predictions] == [1, 0]


def test_score_to_probability_accepts_probabilities_and_logits():
    assert _score_to_probability([0.25]) == 0.25
    assert round(_score_to_probability(2.0), 4) == 0.8808


def test_balance_training_rows_oversamples_positives():
    rows = [
        {"id": "p1", "question_id": "q1", "label": 1},
        {"id": "n1", "question_id": "q1", "label": 0},
        {"id": "n2", "question_id": "q1", "label": 0},
        {"id": "n3", "question_id": "q1", "label": 0},
    ]

    balanced = _balance_training_rows(rows)

    assert sum(int(row["label"] == 1) for row in balanced) == 3
    assert sum(int(row["label"] == 0) for row in balanced) == 3


def test_classification_metrics_counts_confusion_matrix():
    metrics = _classification_metrics(
        labels=[1, 1, 0, 0],
        probabilities=[0.9, 0.2, 0.8, 0.1],
        threshold=0.5,
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_best_threshold_prefers_validation_f1():
    threshold = _best_threshold(
        labels=[1, 1, 0, 0],
        probabilities=[0.9, 0.8, 0.4, 0.1],
    )

    assert 0.4 < threshold <= 0.8
