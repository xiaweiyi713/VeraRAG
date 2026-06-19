"""Tests for CycloneDX SBOM generation."""

import json
from pathlib import Path

from experiments.generate_sbom import build_sbom, main, validate_sbom, write_sbom


def test_sbom_generation_includes_project_and_optional_dependencies(tmp_path):
    _write_pyproject(tmp_path)

    sbom = build_sbom(tmp_path)

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert sbom["metadata"]["component"]["name"] == "verarag"
    component_names = [component["name"] for component in sbom["components"]]
    assert component_names == ["fastapi", "numpy", "pytest", "sentence-transformers"]
    fastapi = _component(sbom, "fastapi")
    assert fastapi["scope"] == "required"
    pytest = _component(sbom, "pytest")
    assert pytest["scope"] == "optional"
    assert {"name": "verarag:dependency-groups", "value": "dev"} in pytest["properties"]


def test_sbom_validation_accepts_fresh_generated_file(tmp_path):
    _write_pyproject(tmp_path)
    output = tmp_path / "build/sbom/verarag-sbom.cdx.json"

    write_sbom(build_sbom(tmp_path), output)
    audit = validate_sbom(output, tmp_path)

    assert audit.valid
    assert audit.errors == []
    assert audit.component_count == 4


def test_sbom_validation_rejects_stale_dependency_components(tmp_path):
    _write_pyproject(tmp_path)
    output = tmp_path / "sbom.json"
    sbom = build_sbom(tmp_path)
    sbom["components"] = sbom["components"][:-1]
    write_sbom(sbom, output)

    audit = validate_sbom(output, tmp_path)

    assert not audit.valid
    assert "SBOM dependency components are stale or incomplete" in audit.errors


def test_sbom_cli_writes_and_validates_json(tmp_path, capsys):
    _write_pyproject(tmp_path)
    output = tmp_path / "sbom.json"

    main(["--project-root", str(tmp_path), "--output", str(output), "--check", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["component_count"] == 4
    assert output.is_file()


def test_release_check_runs_sbom_gate():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "sbom-check:" in makefile
    assert "experiments/generate_sbom.py --output build/sbom/verarag-sbom.cdx.json --check" in makefile
    assert (
        "release-check: lint version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check deps-check "
        "metadata-check sbom-check coverage-check benchmark-check "
        "release-artifacts-check package-check release-checksums-check"
    ) in makefile


def _component(sbom: dict[str, object], name: str) -> dict[str, object]:
    components = sbom["components"]
    assert isinstance(components, list)
    for component in components:
        if isinstance(component, dict) and component.get("name") == name:
            return component
    raise AssertionError(f"missing component {name}")


def _write_pyproject(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "\n".join([
            "[project]",
            'name = "verarag"',
            'version = "0.1.0"',
            "dependencies = [",
            '    "numpy>=1.21.0",',
            '    "fastapi>=0.104.0",',
            "]",
            "",
            "[project.optional-dependencies]",
            'dense = ["sentence-transformers>=2.2.0"]',
            'dev = ["pytest>=7.0.0"]',
            'all = ["verarag[dense,dev]"]',
            "",
            "[project.urls]",
            'Repository = "https://github.com/example/verarag"',
        ])
        + "\n",
        encoding="utf-8",
    )
