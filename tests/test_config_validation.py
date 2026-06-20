"""Tests for YAML configuration validation."""

import json
from pathlib import Path

import yaml

from experiments.validate_configs import main, validate_configs


def test_config_validation_accepts_repository_configs():
    audit = validate_configs()

    assert audit.valid
    assert audit.errors == []
    assert "configs/model.yaml" in audit.files
    assert "configs/deepseek_run.yaml" in audit.files
    assert "configs/deepseek_rules_only.yaml" in audit.files
    assert "configs/verabench_v112_canonical.yaml" in audit.files
    assert "configs/verabench_v112_retrieval_adaptive.yaml" in audit.files
    assert "configs/verabench_v112_retrieval_adaptive_top3.yaml" in audit.files
    assert "configs/verabench_v112_retrieval_rerank_top3.yaml" in audit.files
    assert "configs/verabench_v112_retrieval_rerank_top3_guarded.yaml" in audit.files


def test_canonical_verabench_config_freezes_authoritative_run_identity():
    config = yaml.safe_load(
        Path("configs/verabench_v112_canonical.yaml").read_text(encoding="utf-8")
    )

    assert config["canonical_run"]["name"] == "verabench_v112_canonical_deepseek"
    assert config["canonical_run"]["benchmark_version"] == "1.1.2"
    assert (
        config["canonical_run"]["output"]
        == "outputs/remote_results/verabench_v112_canonical_deepseek.json"
    )
    assert config["canonical_run"]["question_types"] == "all"
    assert config["canonical_run"]["bootstrap"]["seed"] == 1729
    assert config["canonical_run"]["bootstrap"]["resamples"] == 2000
    assert config["llm"]["provider"] == "deepseek"
    assert config["llm"]["model"] == "deepseek-v4-flash"
    assert config["llm"]["temperature"] == 0.0
    assert config["retriever"]["type"] == "bm25"
    assert config["retriever"]["top_k_policy"] == "fixed"
    assert config["pipeline"]["max_retrieval_rounds"] == 1


def test_retrieval_adaptive_config_preserves_canonical_except_policy():
    canonical = yaml.safe_load(
        Path("configs/verabench_v112_canonical.yaml").read_text(encoding="utf-8")
    )
    adaptive = yaml.safe_load(
        Path("configs/verabench_v112_retrieval_adaptive.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert adaptive["canonical_run"]["name"] == "verabench_v112_retrieval_adaptive_deepseek"
    assert adaptive["canonical_run"]["benchmark_version"] == canonical["canonical_run"]["benchmark_version"]
    assert adaptive["llm"] == canonical["llm"]
    assert adaptive["pipeline"] == canonical["pipeline"]
    assert adaptive["retriever"]["type"] == canonical["retriever"]["type"]
    assert adaptive["retriever"]["top_k_policy"] == "complexity_adaptive"
    assert canonical["retriever"]["top_k_policy"] == "fixed"


def test_retrieval_adaptive_top3_config_matches_offline_best_candidate():
    canonical = yaml.safe_load(
        Path("configs/verabench_v112_canonical.yaml").read_text(encoding="utf-8")
    )
    top3 = yaml.safe_load(
        Path("configs/verabench_v112_retrieval_adaptive_top3.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert (
        top3["canonical_run"]["name"]
        == "verabench_v112_retrieval_adaptive_top3_deepseek"
    )
    assert top3["canonical_run"]["benchmark_version"] == canonical["canonical_run"]["benchmark_version"]
    assert top3["llm"] == canonical["llm"]
    assert top3["pipeline"] == canonical["pipeline"]
    assert top3["retriever"]["type"] == canonical["retriever"]["type"]
    assert top3["retriever"]["retrieval_top_k"] == 3
    assert top3["retriever"]["top_k_policy"] == "complexity_adaptive"
    assert top3["retriever"]["adaptive_simple_top_k"] == 2
    assert top3["retriever"]["adaptive_medium_top_k"] == 4
    assert top3["retriever"]["adaptive_complex_top_k"] == 5
    assert "retrieval_top_k" not in canonical["retriever"]


def test_retrieval_rerank_top3_config_matches_offline_frontier_candidate():
    canonical = yaml.safe_load(
        Path("configs/verabench_v112_canonical.yaml").read_text(encoding="utf-8")
    )
    rerank = yaml.safe_load(
        Path("configs/verabench_v112_retrieval_rerank_top3.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert (
        rerank["canonical_run"]["name"]
        == "verabench_v112_retrieval_rerank_top3_deepseek"
    )
    assert rerank["canonical_run"]["benchmark_version"] == canonical["canonical_run"]["benchmark_version"]
    assert rerank["llm"] == canonical["llm"]
    assert rerank["pipeline"] == canonical["pipeline"]
    assert rerank["retriever"]["type"] == "bm25_rerank"
    assert rerank["retriever"]["retrieval_top_k"] == 3
    assert rerank["retriever"]["top_k_policy"] == "complexity_adaptive"
    assert rerank["retriever"]["reranker_model_name"] == "BAAI/bge-reranker-base"
    assert rerank["retriever"]["reranker_candidate_k"] == 5
    assert rerank["retriever"]["reranker_batch_size"] == 16
    assert rerank["retriever"]["reranker_local_files_only"] is False


def test_retrieval_rerank_top3_guarded_config_preserves_base_recall_anchor():
    canonical = yaml.safe_load(
        Path("configs/verabench_v112_canonical.yaml").read_text(encoding="utf-8")
    )
    guarded = yaml.safe_load(
        Path("configs/verabench_v112_retrieval_rerank_top3_guarded.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert (
        guarded["canonical_run"]["name"]
        == "verabench_v112_retrieval_rerank_top3_guarded_deepseek"
    )
    assert guarded["canonical_run"]["benchmark_version"] == canonical["canonical_run"]["benchmark_version"]
    assert guarded["llm"] == canonical["llm"]
    assert guarded["pipeline"] == canonical["pipeline"]
    assert guarded["retriever"]["type"] == "bm25_rerank"
    assert guarded["retriever"]["retrieval_top_k"] == 3
    assert guarded["retriever"]["top_k_policy"] == "complexity_adaptive"
    assert guarded["retriever"]["reranker_candidate_k"] == 5
    assert guarded["retriever"]["reranker_preserve_base_top_k"] == 1


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
                "  retrieval_top_k: 0",
                "  top_k_policy: vibes",
                "  reranker_preserve_base_top_k: -1",
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
    assert (
        "retriever.type",
        "retriever.type must be one of bm25, bm25_rerank, dense, dense_rerank, "
        "hybrid, hybrid_rerank",
    ) in messages
    assert ("retriever.retrieval_top_k", "retriever.retrieval_top_k must be a positive integer") in messages
    assert (
        "retriever.top_k_policy",
        "retriever.top_k_policy must be one of complexity_adaptive, fixed, precision_cap",
    ) in messages
    assert (
        "retriever.reranker_preserve_base_top_k",
        "retriever.reranker_preserve_base_top_k must be a non-negative integer",
    ) in messages
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
