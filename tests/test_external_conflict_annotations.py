"""Tests for independently annotated external conflict-set audits."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.benchmark.external_annotations import (
    audit_external_conflict_set,
    build_external_annotation_packet,
    compile_external_annotation_packet,
)


def test_external_conflict_fixture_passes_protocol_audit():
    report = audit_external_conflict_set(
        "data/external/conflict_mini_v1",
        internal_data_dir="data/verabench",
        min_questions=6,
    )

    assert report["valid"] is True
    assert report["schema_version"] == "external-conflict-annotation-v1"
    assert report["coverage"]["questions"] == 6
    assert report["coverage"]["annotations"] == 12
    assert report["coverage"]["adjudications"] == 6
    assert report["coverage"]["gold_conflict_questions"] == 4
    assert report["agreement"]["conflict_present"]["cohen_kappa"] == 1.0
    assert report["agreement"]["conflict_type"]["comparisons"] == 6
    assert report["agreement"]["disagreement_questions"] == ["EXT006"]
    assert report["adjudication"]["overrides"] == 1
    assert len(report["dataset"]["fingerprints"]["questions.jsonl"]) == 64
    assert (
        report["dataset"]["fingerprints"]["questions.jsonl"]
        != report["dataset"]["internal_fingerprints"]["questions_sha256"]
    )


def test_external_conflict_audit_rejects_insufficient_kappa():
    report = audit_external_conflict_set(
        "data/external/conflict_mini_v1",
        min_conflict_kappa=1.1,
    )

    assert report["valid"] is False
    assert any("kappa" in error for error in report["errors"])


def test_external_conflict_cli_writes_report(tmp_path):
    output_path = tmp_path / "external_audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_external_conflict_set.py",
            "--data-dir",
            "data/external/conflict_mini_v1",
            "--min-questions",
            "6",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == saved
    assert payload["valid"] is True
    assert payload["dataset"]["dataset_id"] == "conflict-mini-v1"


def test_external_conflict_cli_rejects_too_small_dataset():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_external_conflict_set.py",
            "--data-dir",
            "data/external/conflict_mini_v1",
            "--min-questions",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "requires at least 7" in result.stdout


def test_build_external_annotation_packet_is_blind(tmp_path):
    packet = build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        tmp_path,
        annotator_ids=["ann_a", "ann_b"],
    )

    assert packet["schema_version"] == "external-conflict-annotation-packet-v1"
    assert packet["questions"] == 6
    assert packet["template_files"] == {
        "ann_a": "annotations_ann_a.jsonl",
        "ann_b": "annotations_ann_b.jsonl",
    }
    assert "ground_truth_answer" in packet["blind_fields_omitted"]
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "adjudications_template.jsonl").exists()

    rows = [
        json.loads(line)
        for line in (tmp_path / "annotations_ann_a.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 6
    assert rows[0]["annotator_id"] == "ann_a"
    assert rows[0]["label"]["conflict_present"] is None
    serialized = json.dumps(rows, ensure_ascii=False)
    assert "ground_truth_answer" not in serialized
    assert "expected_conflicts" not in serialized
    assert "answer_with_conflict_note" not in serialized
    assert "conflicting" not in serialized
    assert "category" not in serialized


def test_build_external_annotation_packet_rejects_overwrite(tmp_path):
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        tmp_path,
        annotator_ids=["ann_a"],
    )

    with pytest.raises(FileExistsError, match="packet_manifest"):
        build_external_annotation_packet(
            "data/external/conflict_mini_v1",
            tmp_path,
            annotator_ids=["ann_a"],
        )


@pytest.mark.parametrize("annotator_id", ["", " ann_a", "ann_a ", "ann/a", "ann\\a", ".."])
def test_build_external_annotation_packet_rejects_unsafe_annotator_ids(
    tmp_path,
    annotator_id,
):
    with pytest.raises(ValueError, match="annotator ids"):
        build_external_annotation_packet(
            "data/external/conflict_mini_v1",
            tmp_path,
            annotator_ids=[annotator_id],
        )


def test_build_external_annotation_packet_cli(tmp_path):
    output_dir = tmp_path / "packet"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/build_external_annotation_packet.py",
            "--data-dir",
            "data/external/conflict_mini_v1",
            "--output-dir",
            str(output_dir),
            "--annotator",
            "ann_a",
            "--annotator",
            "ann_b",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    saved = json.loads((output_dir / "packet_manifest.json").read_text(encoding="utf-8"))
    assert payload == saved
    assert payload["annotator_ids"] == ["ann_a", "ann_b"]
    assert (output_dir / "annotations_ann_a.jsonl").exists()


def _read_jsonl(path):
    path = Path(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fill_packet_from_fixture(packet_dir):
    labels = {}
    for row in _read_jsonl("data/external/conflict_mini_v1/annotations.jsonl"):
        labels[(row["annotator_id"], row["question_id"])] = row
    for annotator_id in ("ann_a", "ann_b"):
        rows = _read_jsonl(packet_dir / f"annotations_{annotator_id}.jsonl")
        for row in rows:
            label = labels[(annotator_id, row["question_id"])]
            row["label"] = {
                "conflict_present": label["conflict_present"],
                "conflict_type": label["conflict_type"],
                "evidence_pair": label.get("evidence_pair", []),
                "rationale": label["rationale"],
            }
        _write_jsonl(packet_dir / f"annotations_{annotator_id}.jsonl", rows)

    adjudications = {
        row["question_id"]: row
        for row in _read_jsonl("data/external/conflict_mini_v1/adjudications.jsonl")
    }
    rows = _read_jsonl(packet_dir / "adjudications_template.jsonl")
    for row in rows:
        label = adjudications[row["question_id"]]
        row["adjudicator_id"] = label["adjudicator_id"]
        row["gold_label"] = {
            "gold_conflict_present": label["gold_conflict_present"],
            "gold_conflict_type": label["gold_conflict_type"],
            "evidence_pair": label.get("evidence_pair", []),
            "rationale": label["rationale"],
        }
    _write_jsonl(packet_dir / "adjudications_template.jsonl", rows)


def _copy_external_fixture(tmp_path):
    target = tmp_path / "external"
    shutil.copytree("data/external/conflict_mini_v1", target)
    return target


def test_compile_external_annotation_packet_round_trips_to_audit(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )
    _fill_packet_from_fixture(packet_dir)

    compiled = compile_external_annotation_packet(packet_dir, compiled_dir)
    report = audit_external_conflict_set(
        compiled_dir,
        internal_data_dir="data/verabench",
        min_questions=6,
    )

    assert compiled["schema_version"] == "external-conflict-compiled-annotations-v1"
    assert compiled["annotations"] == 12
    assert compiled["adjudications"] == 6
    assert report["valid"] is True
    assert report["agreement"]["conflict_present"]["cohen_kappa"] == 1.0
    assert (compiled_dir / "compiled_manifest.json").exists()


def test_compile_external_annotation_packet_rejects_incomplete_labels(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )

    with pytest.raises(ValueError, match="conflict_present"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_invalid_manifest_schema(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a"],
    )
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "wrong"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=r"packet_manifest\.schema_version"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_non_object_manifest(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    packet_dir.mkdir()
    (packet_dir / "packet_manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_invalid_template_files(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a"],
    )
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["template_files"] = ["annotations_ann_a.jsonl"]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="template_files must be an object"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_missing_label_object(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a"],
    )
    rows = _read_jsonl(packet_dir / "annotations_ann_a.jsonl")
    rows[0].pop("label")
    _write_jsonl(packet_dir / "annotations_ann_a.jsonl", rows)

    with pytest.raises(ValueError, match="label object is required"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_missing_gold_label_object(
    tmp_path,
):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )
    _fill_packet_from_fixture(packet_dir)
    rows = _read_jsonl(packet_dir / "adjudications_template.jsonl")
    rows[0].pop("gold_label")
    _write_jsonl(packet_dir / "adjudications_template.jsonl", rows)

    with pytest.raises(ValueError, match="gold_label object is required"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_supports_packet_relative_source_dir(
    tmp_path,
):
    packet_dir = tmp_path / "packet"
    source_dir = packet_dir / "source"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )
    shutil.copytree("data/external/conflict_mini_v1", source_dir)
    _fill_packet_from_fixture(packet_dir)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_data_dir"] = "source"
    _write_json(manifest_path, manifest)

    compiled = compile_external_annotation_packet(packet_dir, compiled_dir)

    assert compiled["source_data_dir"] == str(source_dir.resolve())
    assert compiled["annotations"] == 12


def test_compile_external_annotation_packet_rejects_missing_packet_source(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a"],
    )
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_data_dir"] = "missing-source"
    _write_json(manifest_path, manifest)

    with pytest.raises(FileNotFoundError, match="source_data_dir"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_template_path_traversal(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )
    _fill_packet_from_fixture(packet_dir)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["template_files"]["ann_a"] = "../annotations_ann_a.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative path markers"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_symlink_escape(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )
    _fill_packet_from_fixture(packet_dir)
    outside = tmp_path / "outside_annotations.jsonl"
    outside.write_text("{}", encoding="utf-8")
    (packet_dir / "annotations_escape.jsonl").symlink_to(outside)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["template_files"]["ann_a"] = "annotations_escape.jsonl"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="outside the packet directory"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_absolute_adjudication_path(
    tmp_path,
):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )
    _fill_packet_from_fixture(packet_dir)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adjudication_template_file"] = str(
        (tmp_path / "adjudications_template.jsonl").resolve()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative to the packet directory"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotation_packet_rejects_existing_outputs(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )
    _fill_packet_from_fixture(packet_dir)
    compiled_dir.mkdir()
    (compiled_dir / "annotations.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError, match="compiled output files already exist"):
        compile_external_annotation_packet(packet_dir, compiled_dir)


def test_compile_external_annotations_cli(tmp_path):
    packet_dir = tmp_path / "packet"
    compiled_dir = tmp_path / "compiled"
    build_external_annotation_packet(
        "data/external/conflict_mini_v1",
        packet_dir,
        annotator_ids=["ann_a", "ann_b"],
    )
    _fill_packet_from_fixture(packet_dir)

    result = subprocess.run(
        [
            sys.executable,
            "experiments/compile_external_annotations.py",
            "--packet-dir",
            str(packet_dir),
            "--output-dir",
            str(compiled_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["annotations"] == 12
    assert (compiled_dir / "annotations.jsonl").exists()


def test_external_conflict_audit_reports_missing_required_files(tmp_path):
    report = audit_external_conflict_set(tmp_path)

    assert report["valid"] is False
    assert len(report["errors"]) == 5
    assert all("missing required file" in error for error in report["errors"])


def test_external_conflict_audit_reports_manifest_and_annotation_errors(tmp_path):
    data_dir = _copy_external_fixture(tmp_path)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("dataset_id")
    manifest.pop("annotation_protocol")
    manifest["schema_version"] = "wrong"
    _write_json(manifest_path, manifest)

    annotations = _read_jsonl(data_dir / "annotations.jsonl")
    annotations.append({**annotations[0]})
    annotations.append({**annotations[0], "question_id": "MISSING"})
    annotations.append({**annotations[0], "annotator_id": ""})
    annotations[0]["conflict_type"] = "unknown_conflict"
    annotations[1]["evidence_pair"] = ["missing", "EXT001_E2"]
    annotations[2]["conflict_type"] = "numeric_conflict"
    annotations[4]["evidence_pair"] = ["EXT003_E1"]
    _write_jsonl(data_dir / "annotations.jsonl", annotations)

    report = audit_external_conflict_set(data_dir)

    joined = "\n".join(report["errors"])
    assert report["valid"] is False
    assert "manifest.schema_version" in joined
    assert "manifest.dataset_id is required" in joined
    assert "manifest.annotation_protocol is required" in joined
    assert "duplicate label" in joined
    assert "unknown question_id 'MISSING'" in joined
    assert "annotator_id is required" in joined
    assert "unknown conflict_type 'unknown_conflict'" in joined
    assert "evidence_pair references unknown evidence id" in joined
    assert "no-conflict labels must use conflict_type 'none'" in joined
    assert "conflict labels require a 2-item evidence_pair" in joined


def test_external_conflict_audit_reports_adjudication_and_coverage_errors(
    tmp_path,
):
    data_dir = _copy_external_fixture(tmp_path)
    annotations = [
        row
        for row in _read_jsonl(data_dir / "annotations.jsonl")
        if row["annotator_id"] == "ann_a" and row["question_id"] != "EXT005"
    ]
    _write_jsonl(data_dir / "annotations.jsonl", annotations)

    adjudications = _read_jsonl(data_dir / "adjudications.jsonl")
    adjudications.append({**adjudications[0]})
    adjudications.append({**adjudications[0], "question_id": "MISSING"})
    adjudications[0]["gold_conflict_type"] = "scope_conflict"
    adjudications[1]["gold_conflict_present"] = True
    adjudications[1]["gold_conflict_type"] = "numeric_conflict"
    adjudications = [
        row for row in adjudications if row["question_id"] != "EXT006"
    ]
    _write_jsonl(data_dir / "adjudications.jsonl", adjudications)

    report = audit_external_conflict_set(data_dir, min_annotators_per_question=2)

    joined = "\n".join(report["errors"])
    assert report["valid"] is False
    assert "not enough paired annotations" in joined
    assert "questions missing annotations: EXT005" in joined
    assert "questions below required annotator count" in joined
    assert "questions missing adjudications: EXT006" in joined
    assert "duplicate adjudication for EXT001" in joined
    assert "adjudications row 7: unknown question_id 'MISSING'" in joined
    assert "gold conflict type 'scope_conflict'" in joined
    assert "gold conflict type 'numeric_conflict'" in joined


def test_external_conflict_audit_rejects_internal_benchmark_fingerprint_match(
    tmp_path,
):
    data_dir = _copy_external_fixture(tmp_path)

    report = audit_external_conflict_set(data_dir, internal_data_dir=data_dir)

    assert report["valid"] is False
    assert "external questions fingerprint matches" in "\n".join(report["errors"])
