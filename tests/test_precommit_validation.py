"""Tests for pre-commit configuration validation."""

import json

from experiments.validate_precommit import main, validate_precommit


def test_precommit_validation_accepts_repository_config():
    audit = validate_precommit()

    assert audit.valid
    assert audit.errors == []
    assert audit.hooks == [
        "configs-check",
        "deployment-check",
        "deps-check",
        "docs-check",
        "doctor-check",
        "examples-check",
        "metadata-check",
        "mypy-public-api",
        "python-support-check",
        "results-check",
        "ruff-check",
        "secret-scan",
        "version-check",
    ]


def test_precommit_validation_reports_missing_hook_and_filename_passing(tmp_path):
    _write_precommit_fixture(
        tmp_path,
        config=(
            "repos:\n"
            "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
            "    hooks:\n"
            "      - id: ruff-check\n"
            "        entry: python -m ruff check verarag src web experiments examples tests configs scripts demo.py demo_local.py run_web.py\n"
            "        language: python\n"
        ),
    )

    audit = validate_precommit(tmp_path)

    assert not audit.valid
    assert ".pre-commit-config.yaml should use local hooks" in audit.errors
    assert ".pre-commit-config.yaml should avoid remote hook repositories" in audit.errors
    assert "pre-commit hook ruff-check should use language: system" in audit.errors
    assert "pre-commit hook ruff-check should set pass_filenames: false" in audit.errors
    assert "pre-commit missing hook docs-check" in audit.errors


def test_precommit_validation_reports_stale_release_and_ci_wiring(tmp_path):
    _write_precommit_fixture(
        tmp_path,
        makefile="release-check: lint docs-check examples-check deployment-check deps-check\n",
        workflow="name: Tests\n",
        contributing="make precommit-check\npre-commit run --all-files\n",
    )

    audit = validate_precommit(tmp_path)

    assert not audit.valid
    assert "Makefile missing precommit-check target" in audit.errors
    assert (
        "release-check should run version-check, python-support-check, doctor-check, and configs-check before docs-check and precommit-check after deployment-check"
        in audit.errors
    )
    assert "GitHub Actions workflow missing Validate pre-commit config step" in audit.errors
    assert "GitHub Actions workflow should run experiments/validate_precommit.py" in audit.errors
    assert "GitHub Actions workflow missing Run environment doctor step" in audit.errors
    assert "GitHub Actions workflow should run experiments/doctor.py" in audit.errors
    assert "GitHub Actions workflow missing Validate default configs step" in audit.errors
    assert "GitHub Actions workflow should run experiments/validate_configs.py" in audit.errors
    assert "CONTRIBUTING missing pre-commit instruction: python -m pip install pre-commit" in audit.errors
    assert "CONTRIBUTING missing pre-commit instruction: pre-commit install" in audit.errors
    assert "CONTRIBUTING missing pre-commit instruction: make configs-check" in audit.errors


def test_precommit_validation_cli_json(capsys):
    exit_code = main(["--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert "ruff-check" in payload["hooks"]


def _write_precommit_fixture(
    root,
    *,
    config: str | None = None,
    makefile: str | None = None,
    pyproject: str | None = None,
    contributing: str | None = None,
    readme: str | None = None,
    workflow: str | None = None,
) -> None:
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".pre-commit-config.yaml").write_text(
        config if config is not None else _valid_precommit_config(),
        encoding="utf-8",
    )
    (root / "Makefile").write_text(
        makefile
        if makefile is not None
        else "precommit-check:\n\tpython experiments/validate_precommit.py\n"
        "release-check: lint version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check deps-check\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        pyproject
        if pyproject is not None
        else 'verarag-validate-precommit = "experiments.validate_precommit:main"\n',
        encoding="utf-8",
    )
    (root / "CONTRIBUTING.md").write_text(
        contributing
        if contributing is not None
        else "python -m pip install pre-commit\npre-commit install\n"
        "pre-commit run --all-files\nmake configs-check\nmake precommit-check\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        readme
        if readme is not None
        else "make configs-check\nmake precommit-check\nverarag-validate-precommit\n",
        encoding="utf-8",
    )
    (root / ".github/workflows/test.yml").write_text(
        workflow
        if workflow is not None
        else "Validate pre-commit config\npython experiments/validate_precommit.py\n",
        encoding="utf-8",
    )


def _valid_precommit_config() -> str:
    return (
        "repos:\n"
        "  - repo: local\n"
        "    hooks:\n"
        "      - id: ruff-check\n"
        "        entry: python -m ruff check verarag src web experiments examples tests configs scripts demo.py demo_local.py run_web.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: mypy-public-api\n"
        "        entry: python -m mypy src/ verarag/ --config-file mypy.ini\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: secret-scan\n"
        "        entry: python experiments/scan_secrets.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: version-check\n"
        "        entry: python experiments/validate_version_identity.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: python-support-check\n"
        "        entry: python experiments/validate_python_support.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: doctor-check\n"
        "        entry: python experiments/doctor.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: configs-check\n"
        "        entry: python experiments/validate_configs.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: docs-check\n"
        "        entry: python experiments/validate_docs.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: results-check\n"
        "        entry: python experiments/validate_results.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: examples-check\n"
        "        entry: python experiments/validate_examples.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: deployment-check\n"
        "        entry: python experiments/validate_deployment.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: deps-check\n"
        "        entry: python experiments/validate_dependency_metadata.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
        "      - id: metadata-check\n"
        "        entry: python experiments/validate_project_metadata.py\n"
        "        language: system\n"
        "        pass_filenames: false\n"
    )
