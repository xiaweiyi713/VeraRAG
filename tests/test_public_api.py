"""Tests for the public verarag package API."""

from pathlib import Path

import verarag
from verarag import (
    VeraBenchEvaluator,
    VeraBenchLoader,
    VeraRAG,
    __version__,
    audit_external_conflict_set,
    audit_verabench_contamination,
    build_conflict_pair_examples,
    build_external_annotation_packet,
    compile_external_annotation_packet,
    load_verabench,
)
from verarag.benchmark import (
    audit_external_conflict_set as audit_external_from_submodule,
)
from verarag.benchmark import (
    audit_verabench_contamination as audit_contamination_from_submodule,
)
from verarag.benchmark import build_conflict_pair_examples as build_pairs_from_submodule
from verarag.benchmark import (
    build_external_annotation_packet as build_packet_from_submodule,
)
from verarag.benchmark import (
    compile_external_annotation_packet as compile_packet_from_submodule,
)
from verarag.benchmark import load_verabench as load_verabench_from_submodule
from verarag.pipeline import VeraRAG as VeraRAGFromSubmodule


def test_public_package_exposes_version():
    assert __version__ == "0.1.0"


def test_public_package_version_uses_distribution_metadata(monkeypatch):
    monkeypatch.setattr(verarag.importlib_metadata, "version", lambda name: "9.9.9")

    assert verarag._read_package_version() == "9.9.9"


def test_public_package_version_falls_back_for_source_checkout(monkeypatch):
    def missing_distribution(name):
        raise verarag.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(verarag.importlib_metadata, "version", missing_distribution)

    assert verarag._read_package_version() == "0.1.0"


def test_public_api_reexports_core_objects():
    from src.benchmark import VeraBenchEvaluator as InternalEvaluator
    from src.benchmark import VeraBenchLoader as InternalLoader
    from src.pipeline.verarag import VeraRAG as InternalVeraRAG

    assert VeraRAG is InternalVeraRAG
    assert VeraRAGFromSubmodule is InternalVeraRAG
    assert VeraBenchLoader is InternalLoader
    assert VeraBenchEvaluator is InternalEvaluator
    assert load_verabench_from_submodule is load_verabench
    assert build_pairs_from_submodule is build_conflict_pair_examples
    assert audit_external_from_submodule is audit_external_conflict_set
    assert audit_contamination_from_submodule is audit_verabench_contamination
    assert build_packet_from_submodule is build_external_annotation_packet
    assert compile_packet_from_submodule is compile_external_annotation_packet


def test_public_api_loads_default_benchmark():
    bench = load_verabench()

    assert len(bench.corpus) == 57
    assert len(bench.questions) == 152


def test_console_scripts_are_registered():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'verarag-web = "web.app:main"' in pyproject
    assert 'verarag-doctor = "experiments.doctor:main"' in pyproject
    assert 'verarag-benchmark = "experiments.run_verabench:main"' in pyproject
    assert 'verarag-analyze = "experiments.analyze_verabench_results:main"' in pyproject
    assert 'verarag-calibration = "experiments.calibration_curve:main"' in pyproject
    assert 'verarag-calibrate-report = "experiments.calibrate_verabench_confidence:main"' in pyproject
    assert 'verarag-leaderboard = "experiments.build_verabench_leaderboard:main"' in pyproject
    assert 'verarag-build-conflict-data = "experiments.build_conflict_training_data:main"' in pyproject
    assert 'verarag-train-conflict = "experiments.train_conflict_cross_encoder:main"' in pyproject
    assert 'verarag-compare-conflicts = "experiments.compare_conflict_detectors:main"' in pyproject
    assert 'verarag-audit-conflict-model = "experiments.audit_conflict_model:main"' in pyproject
    assert 'verarag-conflict-ablation = "experiments.run_conflict_ablation:main"' in pyproject
    assert 'verarag-evaluate-retrieval = "experiments.evaluate_retrieval:main"' in pyproject
    assert 'verarag-plan-retrieval-ablation = "experiments.plan_retrieval_ablation:main"' in pyproject
    assert 'verarag-validate-benchmark = "experiments.validate_verabench:main"' in pyproject
    assert 'verarag-audit-contamination = "experiments.audit_verabench_contamination:main"' in pyproject
    assert 'verarag-build-external-annotation-packet = "experiments.build_external_annotation_packet:main"' in pyproject
    assert 'verarag-compile-external-annotations = "experiments.compile_external_annotations:main"' in pyproject
    assert 'verarag-validate-external-conflicts = "experiments.validate_external_conflict_set:main"' in pyproject
    assert 'verarag-rescore = "experiments.rescore_verabench:main"' in pyproject
    assert 'verarag-merge-reports = "experiments.merge_verabench_reports:main"' in pyproject
    assert 'verarag-scan-secrets = "experiments.scan_secrets:main"' in pyproject
    assert 'verarag-generate-sbom = "experiments.generate_sbom:main"' in pyproject
    assert 'verarag-release-checksums = "experiments.generate_release_checksums:main"' in pyproject
    assert 'verarag-validate-version = "experiments.validate_version_identity:main"' in pyproject
    assert 'verarag-validate-python = "experiments.validate_python_support:main"' in pyproject
    assert 'verarag-validate-configs = "experiments.validate_configs:main"' in pyproject
    assert 'verarag-validate-docs = "experiments.validate_docs:main"' in pyproject
    assert 'verarag-validate-results = "experiments.validate_results:main"' in pyproject
    assert 'verarag-validate-examples = "experiments.validate_examples:main"' in pyproject
    assert 'verarag-validate-deployment = "experiments.validate_deployment:main"' in pyproject
    assert 'verarag-validate-precommit = "experiments.validate_precommit:main"' in pyproject
    assert 'verarag-validate-deps = "experiments.validate_dependency_metadata:main"' in pyproject
    assert 'verarag-validate-metadata = "experiments.validate_project_metadata:main"' in pyproject
    assert 'verarag-validate-package = "experiments.validate_package_contents:main"' in pyproject
    assert 'verarag-validate-install = "experiments.validate_installed_wheel:main"' in pyproject
    assert 'verarag-release-health = "experiments.validate_release_health:main"' in pyproject
