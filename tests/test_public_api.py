"""Tests for the public verarag package API."""

from pathlib import Path

from verarag import VeraBenchEvaluator, VeraBenchLoader, VeraRAG, load_verabench
from verarag.benchmark import load_verabench as load_verabench_from_submodule
from verarag.pipeline import VeraRAG as VeraRAGFromSubmodule


def test_public_api_reexports_core_objects():
    from src.benchmark import VeraBenchEvaluator as InternalEvaluator
    from src.benchmark import VeraBenchLoader as InternalLoader
    from src.pipeline.verarag import VeraRAG as InternalVeraRAG

    assert VeraRAG is InternalVeraRAG
    assert VeraRAGFromSubmodule is InternalVeraRAG
    assert VeraBenchLoader is InternalLoader
    assert VeraBenchEvaluator is InternalEvaluator
    assert load_verabench_from_submodule is load_verabench


def test_public_api_loads_default_benchmark():
    bench = load_verabench()

    assert len(bench.corpus) == 57
    assert len(bench.questions) == 152


def test_console_scripts_are_registered():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'verarag-web = "web.app:main"' in pyproject
    assert 'verarag-benchmark = "experiments.run_verabench:main"' in pyproject
    assert 'verarag-analyze = "experiments.analyze_verabench_results:main"' in pyproject
    assert 'verarag-calibration = "experiments.calibration_curve:main"' in pyproject
    assert 'verarag-leaderboard = "experiments.build_verabench_leaderboard:main"' in pyproject
