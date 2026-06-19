"""Tests for YAML configuration validation."""

import json

from experiments.validate_configs import main, validate_configs


def test_config_validation_accepts_repository_configs():
    audit = validate_configs()

    assert audit.valid
    assert audit.errors == []
    assert "configs/model.yaml" in audit.files
    assert "configs/deepseek_run.yaml" in audit.files
    assert "configs/deepseek_rules_only.yaml" in audit.files


def test_config_validation_reports_runtime_shape_errors(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        "\n".join(
            [
                "llm:",
                "  api_key: real-secret-value",
                "  max_tokens: 0",
                "  model: 123",
                "  provider: deepseek",
                "pipeline:",
                "  max_retrieval_rounds: 0",
                "  enable_repair: sometimes",
                "retriever:",
                "  type: graph",
                "conflict_graph:",
                "  nli_threshold: 1.5",
            ]
        ),
        encoding="utf-8",
    )

    audit = validate_configs([config])

    messages = {(issue.field, issue.message) for issue in audit.errors}
    assert not audit.valid
    assert ("llm.api_key", "llm.api_key must reference an environment variable like ${DEEPSEEK_API_KEY}") in messages
    assert ("llm.max_tokens", "llm.max_tokens must be a positive integer") in messages
    assert ("llm.model", "llm.model must be a non-empty string") in messages
    assert ("pipeline.max_retrieval_rounds", "pipeline.max_retrieval_rounds must be a positive integer") in messages
    assert ("pipeline.enable_repair", "pipeline.enable_repair must be a boolean") in messages
    assert ("retriever.type", "retriever.type must be one of bm25, hybrid, dense") in messages
    assert ("conflict_graph.nli_threshold", "conflict_graph.nli_threshold must be a probability in [0, 1]") in messages


def test_config_validation_reports_dataset_shape_errors(tmp_path):
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: 123",
                "evaluation:",
                "  metrics: []",
                "scoring:",
                "  pass_threshold: 2",
            ]
        ),
        encoding="utf-8",
    )

    audit = validate_configs([config])

    messages = {(issue.field, issue.message) for issue in audit.errors}
    assert not audit.valid
    assert ("dataset.name", "dataset.name must be a non-empty string") in messages
    assert ("dataset.path", "dataset.path is required") in messages
    assert ("evaluation.metrics", "evaluation.metrics must be a non-empty list of strings") in messages
    assert ("scoring.pass_threshold", "scoring.pass_threshold must be a probability in [0, 1]") in messages


def test_config_validation_reports_yaml_parse_failure(tmp_path):
    config = tmp_path / "broken.yaml"
    config.write_text("llm: [unterminated\n", encoding="utf-8")

    audit = validate_configs([config])

    assert not audit.valid
    assert audit.errors[0].field == ""
    assert audit.errors[0].message.startswith("YAML parse failed:")


def test_config_validation_cli_json(capsys):
    exit_code = main(["--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert "configs/model.yaml" in payload["files"]
