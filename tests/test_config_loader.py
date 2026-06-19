"""Tests for config loading helpers."""

import pytest

import configs
from configs import get_dataset_config, load_config, merge_configs


def test_load_config_rejects_path_traversal_and_non_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr(configs, "CONFIG_DIR", tmp_path)

    with pytest.raises(ValueError, match=r"relative \.yaml"):
        load_config("/tmp/model.yaml")

    with pytest.raises(ValueError, match="stay inside configs"):
        load_config("../secret.yaml")

    with pytest.raises(ValueError, match=r"relative \.yaml"):
        load_config("model.json")


def test_load_config_requires_mapping_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr(configs, "CONFIG_DIR", tmp_path)
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    (tmp_path / "list.yaml").write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config("empty.yaml")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config("list.yaml")


def test_merge_configs_deep_copies_nested_values():
    base = {
        "llm": {"provider": "openai", "models": ["a"]},
        "pipeline": {"max_retrieval_rounds": 2},
    }
    override = {"llm": {"provider": "deepseek"}}

    merged = merge_configs(base, override)
    merged["llm"]["models"].append("b")
    merged["pipeline"]["max_retrieval_rounds"] = 9

    assert merged["llm"]["provider"] == "deepseek"
    assert base["llm"]["models"] == ["a"]
    assert base["pipeline"]["max_retrieval_rounds"] == 2


def test_get_dataset_config_accepts_name_or_yaml_suffix(monkeypatch, tmp_path):
    monkeypatch.setattr(configs, "CONFIG_DIR", tmp_path)
    (tmp_path / "model.yaml").write_text(
        "llm:\n  provider: deepseek\npipeline:\n  max_retrieval_rounds: 2\n",
        encoding="utf-8",
    )
    (tmp_path / "demo.yaml").write_text(
        "dataset:\n  name: demo\npipeline:\n  max_retrieval_rounds: 3\n",
        encoding="utf-8",
    )

    assert get_dataset_config("demo")["pipeline"]["max_retrieval_rounds"] == 3
    assert get_dataset_config("demo.yaml")["dataset"]["name"] == "demo"
