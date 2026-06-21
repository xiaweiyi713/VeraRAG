#!/usr/bin/env python3
"""Validate VeraRAG YAML configuration files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ENV_PLACEHOLDER_RE = re.compile(r"^\$\{[A-Z][A-Z0-9_]*\}$")
RUNTIME_CONFIG_NAMES = {
    "model.yaml",
    "deepseek_run.yaml",
    "deepseek_rules_only.yaml",
    "verabench_v112_canonical.yaml",
}
RETRIEVER_TYPES = {
    "bm25",
    "hybrid",
    "dense",
    "bm25_rerank",
    "hybrid_rerank",
    "dense_rerank",
}
TOP_K_POLICIES = {"fixed", "precision_cap", "complexity_adaptive"}
CONFIDENCE_BEHAVIORS = {
    "abstain",
    "answer_with_citation",
    "answer_with_conflict_note",
    "correct_premise",
}


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "field": self.field, "message": self.message}


@dataclass(frozen=True)
class ConfigAudit:
    valid: bool
    errors: list[ConfigIssue]
    warnings: list[ConfigIssue]
    files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "files": self.files,
        }


def validate_configs(
    paths: list[str | Path] | None = None,
    *,
    config_dir: str | Path = "configs",
) -> ConfigAudit:
    root = Path(config_dir)
    config_paths = _resolve_config_paths(paths, root)
    errors: list[ConfigIssue] = []
    warnings: list[ConfigIssue] = []

    if not config_paths:
        errors.append(ConfigIssue(str(root), "", "no YAML config files found"))

    for path in config_paths:
        display_path = path.as_posix()
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            errors.append(ConfigIssue(display_path, "", f"YAML parse failed: {exc}"))
            continue

        if not isinstance(payload, dict):
            errors.append(ConfigIssue(display_path, "", "config root must be a mapping"))
            continue

        is_runtime = path.name in RUNTIME_CONFIG_NAMES or any(
            section in payload for section in ("llm", "retriever", "conflict_graph")
        )
        if is_runtime:
            _validate_runtime_config(display_path, payload, errors)
        else:
            _validate_dataset_config(display_path, payload, errors, warnings)
        _validate_common_sections(display_path, payload, errors)

    return ConfigAudit(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        files=[path.as_posix() for path in config_paths],
    )


def _resolve_config_paths(paths: list[str | Path] | None, config_dir: Path) -> list[Path]:
    if paths:
        return sorted(Path(path) for path in paths)
    if not config_dir.is_dir():
        return []
    return sorted(path for path in config_dir.glob("*.yaml") if path.is_file())


def _validate_runtime_config(
    path: str,
    payload: dict[str, Any],
    errors: list[ConfigIssue],
) -> None:
    llm = _mapping_section(path, payload, "llm", errors, required=True)
    if llm is not None:
        _string(path, "llm.provider", llm.get("provider"), errors, required=True)
        _string(path, "llm.model", llm.get("model"), errors, required=True)
        _env_placeholder(path, "llm.api_key", llm.get("api_key"), errors)
        _string(path, "llm.api_key_env", llm.get("api_key_env"), errors, required=False)
        _string(path, "llm.base_url", llm.get("base_url"), errors, required=False)
        _number_range(path, "llm.temperature", llm.get("temperature"), errors, minimum=0, maximum=2)
        _positive_int(path, "llm.max_tokens", llm.get("max_tokens"), errors, required=True)

    retriever = _mapping_section(path, payload, "retriever", errors, required=True)
    if retriever is not None:
        retriever_type = retriever.get("type")
        if retriever_type not in RETRIEVER_TYPES:
            errors.append(
                ConfigIssue(
                    path,
                    "retriever.type",
                    "retriever.type must be one of bm25, bm25_rerank, dense, "
                    "dense_rerank, hybrid, hybrid_rerank",
                )
            )
        _positive_int(path, "retriever.top_k", retriever.get("top_k"), errors, required=False)
        _positive_int(path, "retriever.fetch_k", retriever.get("fetch_k"), errors, required=False)
        _positive_int(
            path,
            "retriever.retrieval_top_k",
            retriever.get("retrieval_top_k"),
            errors,
            required=False,
        )
        _choice(
            path,
            "retriever.top_k_policy",
            retriever.get("top_k_policy"),
            TOP_K_POLICIES,
            errors,
        )
        for field in (
            "precision_cap_top_k",
            "adaptive_simple_top_k",
            "adaptive_medium_top_k",
            "adaptive_complex_top_k",
            "reranker_candidate_k",
            "reranker_batch_size",
            "targeted_second_pass_top_k",
            "targeted_second_pass_max_new_evidence",
        ):
            _positive_int(path, f"retriever.{field}", retriever.get(field), errors, required=False)
        _boolean(
            path,
            "retriever.targeted_second_pass_enabled",
            retriever.get("targeted_second_pass_enabled"),
            errors,
        )
        _probability(
            path,
            "retriever.targeted_second_pass_coverage_threshold",
            retriever.get("targeted_second_pass_coverage_threshold"),
            errors,
        )
        _non_negative_int(
            path,
            "retriever.reranker_preserve_base_top_k",
            retriever.get("reranker_preserve_base_top_k"),
            errors,
        )
        _string(
            path,
            "retriever.reranker_model_name",
            retriever.get("reranker_model_name"),
            errors,
            required=False,
        )
        _string(path, "retriever.reranker_device", retriever.get("reranker_device"), errors, required=False)
        _boolean(
            path,
            "retriever.reranker_local_files_only",
            retriever.get("reranker_local_files_only"),
            errors,
        )
        _probability(path, "retriever.sparse_weight", retriever.get("sparse_weight"), errors)
        _probability(path, "retriever.dense_weight", retriever.get("dense_weight"), errors)

    _mapping_section(path, payload, "pipeline", errors, required=True)
    reasoning = _mapping_section(path, payload, "reasoning", errors, required=False)
    if reasoning is not None:
        _boolean(
            path,
            "reasoning.enforce_answer_citations",
            reasoning.get("enforce_answer_citations"),
            errors,
        )
        _boolean(
            path,
            "reasoning.claim_slot_selection_enabled",
            reasoning.get("claim_slot_selection_enabled"),
            errors,
        )
        _positive_int(
            path,
            "reasoning.claim_slot_max_evidence",
            reasoning.get("claim_slot_max_evidence"),
            errors,
            required=False,
        )

    uncertainty = _mapping_section(path, payload, "uncertainty", errors, required=False)
    if uncertainty is not None:
        runtime_calibration = _mapping_section(
            path,
            uncertainty,
            "runtime_confidence_calibration",
            errors,
            required=False,
        )
        if runtime_calibration is not None:
            _boolean(
                path,
                "uncertainty.runtime_confidence_calibration.enabled",
                runtime_calibration.get("enabled"),
                errors,
            )
            _probability(
                path,
                "uncertainty.runtime_confidence_calibration.blend_weight",
                runtime_calibration.get("blend_weight"),
                errors,
            )
            _probability(
                path,
                "uncertainty.runtime_confidence_calibration.max_adjustment",
                runtime_calibration.get("max_adjustment"),
                errors,
            )
            behavior_priors = _mapping_section(
                path,
                runtime_calibration,
                "behavior_priors",
                errors,
                required=False,
            )
            if behavior_priors is not None:
                for behavior, prior in behavior_priors.items():
                    if behavior not in CONFIDENCE_BEHAVIORS:
                        errors.append(
                            ConfigIssue(
                                path,
                                f"uncertainty.runtime_confidence_calibration.behavior_priors.{behavior}",
                                "behavior prior key must be one of "
                                f"{', '.join(sorted(CONFIDENCE_BEHAVIORS))}",
                            )
                        )
                    _probability(
                        path,
                        f"uncertainty.runtime_confidence_calibration.behavior_priors.{behavior}",
                        prior,
                        errors,
                    )


def _validate_dataset_config(
    path: str,
    payload: dict[str, Any],
    errors: list[ConfigIssue],
    warnings: list[ConfigIssue],
) -> None:
    dataset = _mapping_section(path, payload, "dataset", errors, required=True)
    if dataset is not None:
        _string(path, "dataset.name", dataset.get("name"), errors, required=True)
        _string(path, "dataset.path", dataset.get("path"), errors, required=True)
        _string_list(path, "dataset.labels", dataset.get("labels"), errors, required=False)
    else:
        warnings.append(ConfigIssue(path, "dataset", "non-runtime config has no dataset section"))

    evaluation = _mapping_section(path, payload, "evaluation", errors, required=False)
    if evaluation is not None:
        _string_list(path, "evaluation.metrics", evaluation.get("metrics"), errors, required=True)

    scoring = _mapping_section(path, payload, "scoring", errors, required=False)
    if scoring is not None:
        _probability(path, "scoring.pass_threshold", scoring.get("pass_threshold"), errors)


def _validate_common_sections(
    path: str,
    payload: dict[str, Any],
    errors: list[ConfigIssue],
) -> None:
    pipeline = _mapping_section(path, payload, "pipeline", errors, required=False)
    if pipeline is not None:
        _positive_int(
            path,
            "pipeline.max_retrieval_rounds",
            pipeline.get("max_retrieval_rounds"),
            errors,
            required=False,
        )
        _positive_int(
            path,
            "pipeline.max_subquestions",
            pipeline.get("max_subquestions"),
            errors,
            required=False,
        )
        for field in (
            "enable_conflict_graph",
            "enable_uncertainty",
            "enable_verification",
            "enable_repair",
            "force_counter_evidence",
        ):
            _boolean(path, f"pipeline.{field}", pipeline.get(field), errors)

    evidence = _mapping_section(path, payload, "evidence", errors, required=False)
    if evidence is not None:
        _boolean(path, "evidence.semantic_dedup", evidence.get("semantic_dedup"), errors)
        _boolean(
            path,
            "evidence.semantic_dedup_local_files_only",
            evidence.get("semantic_dedup_local_files_only"),
            errors,
        )
        _string(
            path,
            "evidence.semantic_dedup_model",
            evidence.get("semantic_dedup_model"),
            errors,
            required=False,
        )
        _positive_int(path, "evidence.max_evidence", evidence.get("max_evidence"), errors, required=False)
        _probability(path, "evidence.relevance_threshold", evidence.get("relevance_threshold"), errors)

    verification = _mapping_section(path, payload, "verification", errors, required=False)
    if verification is not None:
        _string(path, "verification.nli_model", verification.get("nli_model"), errors, required=False)
        _boolean(
            path,
            "verification.nli_local_files_only",
            verification.get("nli_local_files_only"),
            errors,
        )
        _boolean(path, "verification.use_nli", verification.get("use_nli"), errors)
        _probability(path, "verification.nli_threshold", verification.get("nli_threshold"), errors)
        _probability(path, "verification.support_threshold", verification.get("support_threshold"), errors)
        _probability(path, "verification.refute_threshold", verification.get("refute_threshold"), errors)
        _string_list(path, "verification.classes", verification.get("classes"), errors, required=False)

    conflict_graph = _mapping_section(path, payload, "conflict_graph", errors, required=False)
    if conflict_graph is not None:
        for field in (
            "enable_learned_detector",
            "learned_require_context",
            "enable_source_reliability_conflict",
            "enable_scope_conflict",
            "enable_granularity_conflict",
            "compare_within_evidence",
            "enable_nli",
            "nli_local_files_only",
            "enable_llm_adjudication",
        ):
            _boolean(path, f"conflict_graph.{field}", conflict_graph.get(field), errors)
        for field in (
            "learned_threshold",
            "learned_candidate_similarity",
            "nli_threshold",
            "min_conflict_similarity",
            "unattributed_conflict_similarity",
            "llm_adjudication_similarity",
        ):
            _probability(path, f"conflict_graph.{field}", conflict_graph.get(field), errors)
        _string(path, "conflict_graph.nli_model", conflict_graph.get("nli_model"), errors, required=False)
        _string(
            path,
            "conflict_graph.learned_model_path",
            conflict_graph.get("learned_model_path"),
            errors,
            required=False,
        )

    task = _mapping_section(path, payload, "task", errors, required=False)
    if task is not None:
        _positive_int(path, "task.max_hops", task.get("max_hops"), errors, required=False)
        _positive_int(
            path,
            "task.max_evidence_per_claim",
            task.get("max_evidence_per_claim"),
            errors,
            required=False,
        )
        for field in (
            "conflict_aware",
            "claim_verification",
            "require_exact_evidence",
            "requires_supporting_facts",
        ):
            _boolean(path, f"task.{field}", task.get(field), errors)


def _mapping_section(
    path: str,
    payload: dict[str, Any],
    section: str,
    errors: list[ConfigIssue],
    *,
    required: bool,
) -> dict[str, Any] | None:
    value = payload.get(section)
    if value is None:
        if required:
            errors.append(ConfigIssue(path, section, f"{section} section is required"))
        return None
    if not isinstance(value, dict):
        errors.append(ConfigIssue(path, section, f"{section} section must be a mapping"))
        return None
    return value


def _boolean(path: str, field: str, value: Any, errors: list[ConfigIssue]) -> None:
    if value is None:
        return
    if not isinstance(value, bool):
        errors.append(ConfigIssue(path, field, f"{field} must be a boolean"))


def _positive_int(
    path: str,
    field: str,
    value: Any,
    errors: list[ConfigIssue],
    *,
    required: bool,
) -> None:
    if value is None:
        if required:
            errors.append(ConfigIssue(path, field, f"{field} is required"))
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(ConfigIssue(path, field, f"{field} must be a positive integer"))


def _non_negative_int(
    path: str,
    field: str,
    value: Any,
    errors: list[ConfigIssue],
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(ConfigIssue(path, field, f"{field} must be a non-negative integer"))


def _number_range(
    path: str,
    field: str,
    value: Any,
    errors: list[ConfigIssue],
    *,
    minimum: float,
    maximum: float,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
        errors.append(ConfigIssue(path, field, f"{field} must be between {minimum:g} and {maximum:g}"))


def _probability(path: str, field: str, value: Any, errors: list[ConfigIssue]) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        errors.append(ConfigIssue(path, field, f"{field} must be a probability in [0, 1]"))


def _choice(
    path: str,
    field: str,
    value: Any,
    choices: set[str],
    errors: list[ConfigIssue],
) -> None:
    if value is None:
        return
    if value not in choices:
        errors.append(
            ConfigIssue(
                path,
                field,
                f"{field} must be one of {', '.join(sorted(choices))}",
            )
        )


def _string(
    path: str,
    field: str,
    value: Any,
    errors: list[ConfigIssue],
    *,
    required: bool,
) -> None:
    if value is None:
        if required:
            errors.append(ConfigIssue(path, field, f"{field} is required"))
        return
    if not isinstance(value, str) or (required and not value.strip()):
        errors.append(ConfigIssue(path, field, f"{field} must be a non-empty string"))


def _string_list(
    path: str,
    field: str,
    value: Any,
    errors: list[ConfigIssue],
    *,
    required: bool,
) -> None:
    if value is None:
        if required:
            errors.append(ConfigIssue(path, field, f"{field} is required"))
        return
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        errors.append(ConfigIssue(path, field, f"{field} must be a non-empty list of strings"))


def _env_placeholder(path: str, field: str, value: Any, errors: list[ConfigIssue]) -> None:
    if value in (None, ""):
        return
    if not isinstance(value, str):
        errors.append(ConfigIssue(path, field, f"{field} must be a string environment placeholder"))
        return
    if not ENV_PLACEHOLDER_RE.fullmatch(value):
        errors.append(
            ConfigIssue(
                path,
                field,
                f"{field} must reference an environment variable like ${{DEEPSEEK_API_KEY}}",
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Specific YAML config files to validate.")
    parser.add_argument("--config-dir", default="configs", help="Directory to scan when no paths are given.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args(argv)

    audit = validate_configs(args.paths or None, config_dir=args.config_dir)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, ensure_ascii=False))
    elif audit.valid:
        print(f"Config validation passed: {len(audit.files)} YAML files.")
        for warning in audit.warnings:
            print(f"WARNING {warning.path} {warning.field}: {warning.message}")
    else:
        print("Config validation failed:")
        for error in audit.errors:
            print(f"- {error.path} {error.field}: {error.message}")
    return 0 if audit.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
