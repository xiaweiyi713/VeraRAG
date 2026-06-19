"""Tests for the VeraRAG environment doctor."""

import json

from experiments import doctor
from experiments.doctor import main, run_doctor


def test_doctor_accepts_repository_state():
    report = run_doctor()

    assert report.valid
    assert "Python runtime meets the supported floor" in report.checks
    assert report.paths["data/verabench/questions.jsonl"]
    assert report.required_modules["numpy"]


def test_doctor_reports_missing_required_paths(tmp_path):
    report = run_doctor(tmp_path)

    assert not report.valid
    assert "required project file missing: pyproject.toml" in report.errors
    assert "required project file missing: data/verabench/questions.jsonl" in report.errors


def test_doctor_reports_optional_missing_without_invalidating(monkeypatch):
    real_module_available = doctor._module_available

    def fake_module_available(module: str) -> bool:
        if module == "sentence_transformers":
            return False
        return real_module_available(module)

    monkeypatch.setattr(doctor, "_module_available", fake_module_available)

    report = run_doctor()

    assert report.valid
    assert not report.optional_features["dense retrieval"]["available"]
    assert "sentence_transformers" in report.optional_features["dense retrieval"]["missing"]
    assert any("optional feature unavailable: dense retrieval" in warning for warning in report.warnings)


def test_doctor_cli_json_and_fail_on_warnings(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)

    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["valid"] is True
    assert payload["llm_environment"]["DEEPSEEK_API_KEY"] is False

    assert main(["--fail-on-warnings"]) == 1
